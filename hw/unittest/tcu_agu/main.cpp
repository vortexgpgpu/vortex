// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_tcu_agu: a TCU_LD of NUM_TCU_LANES metadata words through an LSU client
// port of NUM_LSU_LANES lanes, i.e. in R = TCU / LSU batches. An LSU model
// answers every request with the memory's words, in order, split into two
// fragments, or with the batches reversed. Each TCU lane t must be written
// once, with the word at base + 4t, before the load's result is released.

#include "VVX_tcu_agu_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <deque>
#include <memory>
#include <vector>

#ifndef TCU_LANES
#error "TCU_LANES"
#endif
#ifndef LSU_LANES
#error "LSU_LANES"
#endif
#define BATCHES (TCU_LANES / LSU_LANES)
#define WORD_BITS 64            // LSU word (XLEN 64)
#define ADDR_BITS 45            // LSU word address: 48-bit memory, 8-byte words
#define RATIO (WORD_BITS / 32)

double sc_time_stamp() { return 0; }

static int failures = 0;
#define CHECK(x, ...)                                                      \
  do {                                                                     \
    if (!(x)) {                                                            \
      if (failures < 20) { printf("FAILED: %s: ", #x); printf(__VA_ARGS__); printf("\n"); } \
      ++failures;                                                          \
    }                                                                      \
  } while (false)

// bit access to Verilator's integral and wide signals
template <typename T> static uint64_t getb(const T& v, int lsb, int w) {
  return (uint64_t(v) >> lsb) & ((w == 64) ? ~0ull : ((1ull << w) - 1));
}
template <std::size_t N> static uint64_t getb(const VlWide<N>& v, int lsb, int w) {
  uint64_t r = 0;
  for (int i = 0; i < w; ++i) r |= uint64_t((v[(lsb + i) / 32] >> ((lsb + i) % 32)) & 1) << i;
  return r;
}
template <typename T> static void setb(T& v, int lsb, int w, uint64_t x) {
  T m = T(((w == 64) ? ~0ull : ((1ull << w) - 1))) << lsb;
  v = (v & ~m) | ((T(x) << lsb) & m);
}
template <std::size_t N> static void setb(VlWide<N>& v, int lsb, int w, uint64_t x) {
  for (int i = 0; i < w; ++i) {
    uint32_t& word = v[(lsb + i) / 32];
    uint32_t bit = 1u << ((lsb + i) % 32);
    word = ((x >> i) & 1) ? (word | bit) : (word & ~bit);
  }
}

static uint32_t mem_word(uint64_t byte_addr) {
  return uint32_t(byte_addr * 2654435761u) ^ 0x5a5a0000u ^ uint32_t(byte_addr >> 7);
}

struct Req { uint64_t tag; std::vector<uint64_t> addr; uint32_t mask; };
struct Frag { uint64_t tag; uint32_t mask; bool sop, eop; std::vector<uint64_t> data; };

enum Mode { IN_ORDER = 0, SPLIT = 1, REVERSED = 2 };

static void run(int mode, uint32_t wid, uint32_t slot, uint64_t base) {
  auto ctx = std::make_unique<VerilatedContext>();
  ctx->randReset(2);
  ctx->randSeed(1 + mode);
  ctx->fatalOnError(false);
  auto m = std::make_unique<VVX_tcu_agu_top>(ctx.get());
  auto cycle = [&]() { m->clk = 0; m->eval(); ctx->timeInc(1); m->clk = 1; m->eval(); ctx->timeInc(1); };

  m->ld_valid = 0; m->req_ready = 0; m->rsp_valid = 0; m->result_ready = 0;
  m->reset = 1;
  for (int i = 0; i < 4; ++i) cycle();
  m->reset = 0;
  cycle();

  m->ld_valid = 1; m->ld_wid = wid; m->ld_slot = slot; m->ld_addr = base;
  m->eval();
  CHECK(m->ld_ready, "mode %d: AGU not ready", mode);
  cycle();
  m->ld_valid = 0;

  std::vector<Req> reqs;
  std::deque<Frag> frags;
  std::vector<int> written(TCU_LANES, 0);
  bool result = false;
  int stall = 0;
  for (int t = 0; t < 400 && !result; ++t) {
    // response side: fragments queued in earlier cycles
    m->rsp_valid = 0;
    if (!frags.empty()) {
      auto& f = frags.front();
      m->rsp_valid = 1; m->rsp_mask = f.mask; m->rsp_sop = f.sop; m->rsp_eop = f.eop;
      setb(m->rsp_tag, 0, 16, f.tag);
      for (int i = 0; i < LSU_LANES; ++i) setb(m->rsp_data, i * WORD_BITS, WORD_BITS, f.data[i]);
    }
    // request side: accept every other cycle
    m->req_ready = (t & 1);
    m->eval();
    if (m->req_valid && m->req_ready) {
      Req r;
      r.tag = getb(m->req_tag, 0, 16);
      r.mask = uint32_t(getb(m->req_mask, 0, LSU_LANES));
      CHECK(!m->req_rw, "mode %d: TCU_LD request is a store", mode);
      for (int i = 0; i < LSU_LANES; ++i) r.addr.push_back(getb(m->req_addr, i * ADDR_BITS, ADDR_BITS));
      reqs.push_back(r);
      if (reqs.size() == BATCHES) {  // all batches out: answer them
        std::vector<Req> order = reqs;
        if (mode == REVERSED) std::reverse(order.begin(), order.end());
        for (auto& q : order) {
          Frag f; f.tag = q.tag; f.sop = true; f.eop = true;
          for (int i = 0; i < LSU_LANES; ++i) f.data.push_back(0);
          for (int i = 0; i < LSU_LANES; ++i) {
            uint64_t a = q.addr[i] * 8;
            f.data[i] = uint64_t(mem_word(a)) | (uint64_t(mem_word(a + 4)) << 32);
          }
          if (mode == SPLIT) {
            uint32_t lo = (1u << (LSU_LANES / 2)) - 1;
            Frag a = f, b = f;
            a.mask = q.mask & lo; a.eop = false;
            b.mask = q.mask & ~lo; b.sop = false;
            frags.push_back(a); frags.push_back(b);
          } else {
            f.mask = q.mask;
            frags.push_back(f);
          }
        }
      }
    }
    m->result_ready = 1;
    m->eval();
    if (m->meta_wr_en) {
      CHECK(m->meta_wr_wid == wid && m->meta_wr_idx == slot, "mode %d: write to wid %u slot %u", mode,
            (unsigned)m->meta_wr_wid, (unsigned)m->meta_wr_idx);
      for (int l = 0; l < TCU_LANES; ++l) {
        if (!getb(m->meta_wr_mask, l, 1)) continue;
        ++written[l];
        uint32_t got = uint32_t(getb(m->meta_wr_data, l * 32, 32));
        CHECK(got == mem_word(base + 4ull * l), "mode %d: lane %d got 0x%08x, expected 0x%08x", mode, l, got,
              mem_word(base + 4ull * l));
      }
    }
    if (m->result_valid) {
      result = true;
      CHECK(m->result_wid == wid, "mode %d: result wid %u", mode, (unsigned)m->result_wid);
      for (int l = 0; l < TCU_LANES; ++l)
        CHECK(written[l] == 1, "mode %d: result released with lane %d written %d times", mode, l, written[l]);
    }
    bool rsp_fire = m->rsp_valid && m->rsp_ready;
    cycle();
    if (rsp_fire) frags.pop_front();
    if (m->rsp_valid && !m->rsp_ready) ++stall;
  }
  CHECK(result, "mode %d: no result", mode);
  CHECK(reqs.size() == BATCHES, "mode %d: %zu requests, expected %d", mode, reqs.size(), BATCHES);
  for (size_t b = 0; b < reqs.size(); ++b) {
    CHECK(reqs[b].tag % BATCHES == b, "mode %d: request %zu carries batch %llu", mode, b, (unsigned long long)reqs[b].tag);
    CHECK(reqs[b].mask == ((LSU_LANES == 32) ? ~0u : ((1u << LSU_LANES) - 1)), "mode %d: request %zu mask 0x%x", mode, b, reqs[b].mask);
    for (int i = 0; i < LSU_LANES; ++i) {
      uint64_t t = b * LSU_LANES + i;
      CHECK(reqs[b].addr[i] == (base / 8) + t / RATIO, "mode %d: batch %zu lane %d addr 0x%llx", mode, b, i,
            (unsigned long long)reqs[b].addr[i]);
    }
  }
  // released: the AGU takes the next load
  m->result_ready = 1;
  cycle();
  m->eval();
  CHECK(m->ld_ready && !m->result_valid, "mode %d: not released", mode);
  CHECK(!ctx->gotError(), "mode %d: assertion fired", mode);
  m->final();
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  for (int mode = 0; mode < 3; ++mode) {
    run(mode, 3, 7, 0x80001000ull);
    run(mode, 1, 0, 0x80002468ull & ~7ull);
  }
  if (failures) {
    printf("TCU %d / LSU %d (R=%d): %d check(s) FAILED\n", TCU_LANES, LSU_LANES, BATCHES, failures);
    return 1;
  }
  printf("TCU %d / LSU %d (R=%d): PASSED\n", TCU_LANES, LSU_LANES, BATCHES);
  return 0;
}
