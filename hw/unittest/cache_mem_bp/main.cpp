// Copyright © 2019-2026
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

// A write-back last-level cache against a memory that bounds its reads in
// flight, as every real one does: per port the memory takes a read only while
// it holds fewer than RD_CAP (an AXI master's ID space, a slave's read slots),
// and the request channel is in order, so a read it cannot take holds the
// writes behind it. In stalling mode (the default) the memory also takes no
// request while a response it presents is not taken, as a pipelined memory
// whose output is blocked does; AXI allows a slave's AR/AW ready to depend on
// RREADY. Responses come back out of order, and the cores' response side is
// back-pressured in stretches. The cores stream reads and writes over 64 times
// the cache, so most fills evict a dirty line. The cache must take every fill
// whatever its request side waits on; the test fails on any stall of 20000
// cycles with work pending, on wrong read data, and on a response nobody
// asked for. Arguments: [cycles] [seed] [read cap] [stalling 0|1].

#include "VVX_cache_top.h"
#include "vl_simulator.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#ifndef RD_CAP
#define RD_CAP 16
#endif

#ifndef NREQS
#define NREQS 4
#endif
#ifndef WORD_BYTES
#define WORD_BYTES 16
#endif

static constexpr int NREQ = NREQS, NPORT = 4;
static constexpr int WORD_U32 = WORD_BYTES / 4, WORDS_PER_SECTOR = 64 / WORD_BYTES, WORDS_PER_LINE = 128 / WORD_BYTES;
static constexpr uint32_t FOOT_LINES = 64 * 16384 / 128;
static constexpr uint64_t WATCHDOG = 20000;

using Word = std::array<uint32_t, WORD_U32>;

static uint32_t init_u32(uint64_t word_addr, int k) {
  uint64_t x = 0x9E3779B97F4A7C15ull * (word_addr * 4 + k + 1);
  return uint32_t(x ^ (x >> 31));
}

double sc_time_stamp() { return 0; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t cycles = 100000;
  uint32_t seed = 1;
  if (argc > 1) { long v = std::strtol(argv[1], nullptr, 10); if (v > 0) cycles = v; }
  if (argc > 2) seed = std::strtoul(argv[2], nullptr, 10);
  uint32_t rd_cap = RD_CAP;
  if (argc > 3) rd_cap = std::strtoul(argv[3], nullptr, 10);
  int mode = argc > 4 ? atoi(argv[4]) : 1;
  bool rsp_stalled[NPORT] = {};
  std::mt19937 rng(seed);
  auto pct = [&](int p) { return int(rng() % 100) < p; };

  vl_simulator<VVX_cache_top> sim;

  std::map<uint64_t, Word> mem;  // memory by word address
  auto mem_get = [&](uint64_t wa) {
    auto it = mem.find(wa);
    if (it != mem.end()) return it->second;
    Word w; for (int k = 0; k < WORD_U32; ++k) w[k] = init_u32(wa, k);
    return w;
  };

  struct Core {
    bool valid = false, rw = false;
    uint64_t addr = 0, byteen = 0; uint32_t tag = 0; Word data{};
    std::map<uint64_t, Word> view;
  } core[NREQ];
  std::map<uint32_t, std::pair<int, Word>> reads;  // tag -> (port, expected)
  std::map<uint32_t, int> writes;                  // tag -> port, if the cache answers writes
  uint32_t next_tag = 0;

  struct Rd { uint32_t tag; uint64_t sector; uint64_t ready; std::array<Word, WORDS_PER_SECTOR> data; };
  struct Port { std::vector<Rd> out; bool rsp_valid = false; Rd rsp{}; } port[NPORT];

  for (int i = 0; i < NREQ; ++i) { sim->core_req_valid[i] = 0; sim->core_rsp_ready[i] = 0; }
  for (int p = 0; p < NPORT; ++p) { sim->mem_req_ready[p] = 0; sim->mem_rsp_valid[p] = 0; }
  uint64_t ticks = sim.reset(0);
  for (int i = 0; i < 64; ++i) ticks = sim.step(ticks, 2);  // tag-RAM init

  uint64_t n_fill_bp = 0, n_head_rd_blocked = 0;
  uint64_t n_rd = 0, n_wr = 0, n_rsp = 0, n_fill = 0, n_wb = 0, errors = 0, last_progress = 0, c = 0;
  auto fail = [&](const char* what, int i) {
    if (errors < 16) printf("ERROR cycle %lu: %s (%d)\n", (unsigned long)c, what, i);
    ++errors;
  };

  const uint64_t total = cycles + 200000;
  for (; c < total && errors == 0; ++c) {
    const bool issue = c < cycles;
    for (int i = 0; i < NREQ; ++i) {
      auto& r = core[i];
      if (!r.valid && issue && pct(90) && reads.size() < 4096) {
        // word address whose line belongs to requester i
        uint64_t line = (rng() % FOOT_LINES) & ~uint64_t(NREQ - 1);
        r.addr = (line | i) * WORDS_PER_LINE + rng() % WORDS_PER_LINE;
        r.rw = pct(50);
        const uint64_t all = (WORD_BYTES == 64) ? ~0ull : ((1ull << WORD_BYTES) - 1);
        r.byteen = r.rw ? ((((uint64_t(rng()) << 32) | rng()) & all) | 1) : all;
        for (auto& v : r.data) v = rng();
        r.tag = next_tag++ & 0xffff;
        r.valid = true;
      }
      sim->core_req_valid[i] = r.valid;
      sim->core_req_rw[i] = r.rw;
      sim->core_req_addr[i] = r.addr;
      sim->core_req_byteen[i] = r.byteen;
      sim->core_req_attr[i] = 0;
      sim->core_req_tag[i] = r.tag;
      for (int k = 0; k < WORD_U32; ++k) sim->core_req_data[i][k] = r.data[k];
      sim->core_rsp_ready[i] = pct(((c / 1000) % 2) ? 5 : 90);  // stretches of response back-pressure
    }
    for (int p = 0; p < NPORT; ++p) {
      auto& m = port[p];
      if (!m.rsp_valid) {
        std::vector<size_t> cand;
        for (size_t j = 0; j < m.out.size(); ++j) if (m.out[j].ready <= c) cand.push_back(j);
        if (!cand.empty()) {
          size_t j = cand[rng() % cand.size()];
          m.rsp = m.out[j];
          m.rsp_valid = true;
        }
      }
      sim->mem_rsp_valid[p] = m.rsp_valid;
      sim->mem_rsp_tag[p] = m.rsp.tag;
      for (int w = 0; w < WORDS_PER_SECTOR; ++w)
        for (int k = 0; k < WORD_U32; ++k) sim->mem_rsp_data[p][w * WORD_U32 + k] = m.rsp.data[w][k];
      sim->mem_req_ready[p] = 0;
    }
    sim->eval();
    // the memory takes a read only below its in-flight bound; writes always drain
    bool mreq_fire[NPORT];
    for (int p = 0; p < NPORT; ++p) {
      bool rdy = sim->mem_req_rw[p] ? pct(80) : (port[p].out.size() < rd_cap && pct(80));
      if (mode == 1 && rsp_stalled[p]) rdy = false;
      sim->mem_req_ready[p] = rdy;
    }
    sim->eval();

    bool progress = false;
    for (int p = 0; p < NPORT; ++p) {
      auto& m = port[p];
      mreq_fire[p] = sim->mem_req_valid[p] && sim->mem_req_ready[p];
      if (mreq_fire[p]) {
        progress = true;
        uint64_t sector = sim->mem_req_addr[p];
        if (sim->mem_req_rw[p]) {
          uint64_t be = sim->mem_req_byteen[p];
          for (int w = 0; w < WORDS_PER_SECTOR; ++w) {
            Word d = mem_get(sector * WORDS_PER_SECTOR + w);
            for (int k = 0; k < WORD_U32; ++k) {
              uint32_t v = sim->mem_req_data[p][w * WORD_U32 + k];
              for (int b = 0; b < 4; ++b)
                if ((be >> (w * 16 + k * 4 + b)) & 1) d[k] = (d[k] & ~(0xffu << (8 * b))) | (v & (0xffu << (8 * b)));
            }
            mem[sector * WORDS_PER_SECTOR + w] = d;
          }
          ++n_wb;
        } else {
          Rd rd{uint32_t(sim->mem_req_tag[p]), sector, c + 50 + rng() % 200, {}};
          for (int w = 0; w < WORDS_PER_SECTOR; ++w) rd.data[w] = mem_get(sector * WORDS_PER_SECTOR + w);
          m.out.push_back(rd);
          ++n_fill;
        }
      }
      n_fill_bp += m.rsp_valid && !sim->mem_rsp_ready[p];
      rsp_stalled[p] = m.rsp_valid && !sim->mem_rsp_ready[p];
      n_head_rd_blocked += sim->mem_req_valid[p] && !sim->mem_req_rw[p] && m.out.size() >= rd_cap;
      if (m.rsp_valid && sim->mem_rsp_ready[p]) {
        progress = true;
        for (size_t j = 0; j < m.out.size(); ++j)
          if (m.out[j].tag == m.rsp.tag && m.out[j].sector == m.rsp.sector) { m.out.erase(m.out.begin() + j); break; }
        m.rsp_valid = false;
      }
    }
    for (int i = 0; i < NREQ; ++i) {
      auto& r = core[i];
      if (r.valid && sim->core_req_ready[i]) {
        progress = true;
        auto it = r.view.find(r.addr);
        Word cur;
        if (it != r.view.end()) cur = it->second; else for (int k = 0; k < WORD_U32; ++k) cur[k] = init_u32(r.addr, k);
        if (r.rw) {
          for (int k = 0; k < WORD_U32; ++k)
            for (int b = 0; b < 4; ++b)
              if ((r.byteen >> (k * 4 + b)) & 1ull) cur[k] = (cur[k] & ~(0xffu << (8 * b))) | (r.data[k] & (0xffu << (8 * b)));
          r.view[r.addr] = cur;
          writes[r.tag] = i;
          ++n_wr;
        } else {
          reads[r.tag] = {i, cur};
          ++n_rd;
        }
        r.valid = false;
      }
      if (sim->core_rsp_valid[i] && sim->core_rsp_ready[i]) {
        progress = true;
        uint32_t tag = sim->core_rsp_tag[i];
        auto it = reads.find(tag);
        if (it != reads.end()) {
          if (it->second.first != i) fail("read response on the wrong port", i);
          for (int k = 0; k < WORD_U32; ++k)
            if (uint32_t(sim->core_rsp_data[i][k]) != it->second.second[k]) { fail("wrong read data", i); break; }
          reads.erase(it);
          ++n_rsp;
        } else if (writes.count(tag)) {
          writes.erase(tag);
        } else {
          fail("response with a tag nobody issued", i);
        }
      }
    }

    bool pending = !reads.empty();
    for (int i = 0; i < NREQ; ++i) pending |= core[i].valid;
    for (int p = 0; p < NPORT; ++p) pending |= port[p].rsp_valid || !port[p].out.empty() || sim->mem_req_valid[p];
    if (progress) last_progress = c;
    if (pending && c - last_progress > WATCHDOG) {
      fail("no progress: deadlock", 0);
      for (int p = 0; p < NPORT; ++p)
        printf("  port %d: mem_req valid=%d rw=%d | reads in flight=%zu | rsp valid=%d ready=%d\n", p,
               sim->mem_req_valid[p], sim->mem_req_rw[p], port[p].out.size(), port[p].rsp_valid, sim->mem_rsp_ready[p]);
      for (int i = 0; i < NREQ; ++i)
        printf("  core %d: req valid=%d ready=%d\n", i, core[i].valid, sim->core_req_ready[i]);
    }
    if (!issue && !pending) break;
    ticks = sim.step(ticks, 2);
  }
  if (errors == 0 && !reads.empty()) fail("reads unanswered at the end", 0);
  printf("cache_mem_bp rd_cap=%u seed=%u: cycles=%lu reads=%lu writes=%lu responses=%lu fills=%lu writebacks=%lu fill-backpressure=%lu read-head-blocked=%lu errors=%lu\n%s\n",
         rd_cap, seed, (unsigned long)c, (unsigned long)n_rd, (unsigned long)n_wr, (unsigned long)n_rsp,
         (unsigned long)n_fill, (unsigned long)n_wb, (unsigned long)n_fill_bp, (unsigned long)n_head_rd_blocked, (unsigned long)errors, errors ? "FAILED" : "PASSED");
  return errors ? 1 : 0;
}
