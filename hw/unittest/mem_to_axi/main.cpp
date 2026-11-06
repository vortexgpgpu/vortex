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

// Functional test for VX_mem_to_axi against an adversarial AXI4 slave per
// bank: AR, AW and W ready at random and independently, reads answered out of
// order across IDs (in order within one), writes made visible only when their
// B is sent, and R held off for long stretches. The requesters issue reads and
// writes on a few lines per port, so a read often follows a write to the same
// line still in flight, and the response side is back-pressured at random.
// Checked: every read returns the port's last write before it (the adapter
// orders a read behind an outstanding write to its line), every response tag
// is outstanding and answered once, the master keeps a presented AR/AW/W
// stable until its handshake, and the whole system never stops making
// progress while work is pending.

#include "VVX_mem_to_axi.h"
#include "vl_simulator.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <vector>

#ifndef NI
#define NI 4
#endif
#ifndef NB
#define NB 4
#endif
#ifndef TIDW
#define TIDW 6
#endif

static constexpr int BANK_BITS = (NB > 1) ? __builtin_ctz(NB) : 0;
static constexpr int PORT_BITS = (NI > 1) ? __builtin_ctz(NI) : 0;
static constexpr int LINES_PER_PORT = 8;   // few lines: frequent same-line reuse
static constexpr int NUM_TAGS = 256;       // TAG_WIDTH_IN = 8
static constexpr int MAX_OUTSTANDING = 40; // above the tag buffer's 16
static constexpr uint64_t WATCHDOG = 20000;

static uint64_t init_value(uint32_t addr) {
  uint64_t x = 0x9E3779B97F4A7C15ull * (addr + 1);
  return x ^ (x >> 29);
}

static uint64_t merge(uint64_t old, uint64_t data, uint32_t byteen) {
  uint64_t m = 0;
  for (int i = 0; i < 8; ++i)
    if ((byteen >> i) & 1) m |= 0xffull << (8 * i);
  return (old & ~m) | (data & m);
}

struct Requester {
  bool valid = false;
  bool rw = false;
  uint32_t addr = 0, byteen = 0, tag = 0;
  uint64_t data = 0;
  std::deque<uint32_t> free_tags;
  std::map<uint32_t, uint64_t> expect;  // outstanding read tag -> data
  std::map<uint32_t, uint64_t> view;    // this port's lines as it last wrote them
};

struct Slave {
  struct Rd { uint32_t id; uint64_t data; uint64_t ready; };
  struct Aw { uint32_t id; uint32_t addr; };
  struct Wd { uint64_t data; uint32_t strb; };
  struct Wr { uint32_t id; uint32_t addr; uint64_t data; uint32_t strb; uint64_t ready; };
  std::vector<Rd> reads;
  std::deque<Aw> aws;
  std::deque<Wd> ws;
  std::deque<Wr> writes;
  bool r_valid = false; Rd r_cur{};
  bool b_valid = false; Wr b_cur{};
};

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t cycles = 300000;
  uint32_t seed = 1;
  if (argc > 1) { long v = std::strtol(argv[1], nullptr, 10); if (v > 0) cycles = v; }
  if (argc > 2) seed = std::strtoul(argv[2], nullptr, 10);
  std::mt19937 rng(seed);
  auto pct = [&](int p) { return int(rng() % 100) < p; };

  vl_simulator<VVX_mem_to_axi> sim;
  std::map<uint32_t, uint64_t> mem;  // committed slave memory, by input address
  auto mem_get = [&](uint32_t a) { auto it = mem.find(a); return it == mem.end() ? init_value(a) : it->second; };

  Requester req[NI];
  Slave slv[NB];
  for (int p = 0; p < NI; ++p)
    for (uint32_t t = 0; t < NUM_TAGS; ++t) req[p].free_tags.push_back(t);

  for (int p = 0; p < NI; ++p) { sim->mem_req_valid[p] = 0; sim->mem_rsp_ready[p] = 0; }
  for (int b = 0; b < NB; ++b) {
    sim->m_axi_arready[b] = 0; sim->m_axi_awready[b] = 0; sim->m_axi_wready[b] = 0;
    sim->m_axi_rvalid[b] = 0; sim->m_axi_bvalid[b] = 0;
  }
  uint64_t ticks = sim.reset(0);

  uint64_t n_rd = 0, n_wr = 0, n_rsp = 0, errors = 0, last_progress = 0;
  bool p_ar[NB] = {}, p_aw[NB] = {}, p_w[NB] = {};
  uint64_t p_ar_v[NB] = {}, p_aw_v[NB] = {}, p_w_v[NB] = {};
  auto fail = [&](uint64_t c, const char* what, int i) {
    if (errors < 16) printf("ERROR cycle %lu: %s (%d)\n", (unsigned long)c, what, i);
    ++errors;
  };

  const uint64_t total = cycles + 100000;
  uint64_t c = 0;
  for (; c < total && errors == 0; ++c) {
    const bool issue = c < cycles;
    const int phase = (c / 3000) % 4;  // write-heavy, mixed, read-heavy, stalled responses
    const int p_req = issue ? (phase == 2 ? 70 : 50) : 0;
    const int p_wr  = phase == 0 ? 70 : (phase == 1 ? 35 : 10);
    const int p_rsp = phase == 3 ? 5 : 80;
    const int p_rdy = phase == 3 ? 90 : 60;

    // requesters
    for (int p = 0; p < NI; ++p) {
      auto& r = req[p];
      if (!r.valid && pct(p_req) && !r.free_tags.empty() && r.expect.size() < MAX_OUTSTANDING) {
        uint32_t line = rng() % LINES_PER_PORT;
        uint32_t bank = rng() % NB;
        r.valid = true;
        r.rw = pct(p_wr);
        r.addr = (((line << PORT_BITS) | p) << BANK_BITS) | bank;
        r.byteen = r.rw ? ((rng() & 0xff) | 1) : 0xff;
        r.data = (uint64_t(rng()) << 32) | rng();
        r.tag = r.free_tags.front();
      }
      sim->mem_req_valid[p] = r.valid;
      sim->mem_req_rw[p] = r.rw;
      sim->mem_req_addr[p] = r.addr;
      sim->mem_req_byteen[p] = r.byteen;
      sim->mem_req_data[p] = r.data;
      sim->mem_req_tag[p] = r.tag;
      sim->mem_rsp_ready[p] = pct(p_rsp);
    }

    // slaves
    for (int b = 0; b < NB; ++b) {
      auto& s = slv[b];
      if (!s.r_valid && !s.reads.empty() && pct(phase == 3 ? 20 : 70)) {
        // the oldest read of each id is eligible; take one at random
        std::vector<size_t> cand;
        for (size_t i = 0; i < s.reads.size(); ++i) {
          bool first = true;
          for (size_t j = 0; j < i; ++j) first &= (s.reads[j].id != s.reads[i].id);
          if (first && s.reads[i].ready <= c) cand.push_back(i);
        }
        if (!cand.empty()) {
          size_t k = cand[rng() % cand.size()];
          s.r_cur = s.reads[k];
          s.reads.erase(s.reads.begin() + k);
          s.r_valid = true;
        }
      }
      if (!s.b_valid && !s.writes.empty() && s.writes.front().ready <= c) {
        s.b_cur = s.writes.front();
        s.writes.pop_front();
        s.b_valid = true;
      }
      sim->m_axi_arready[b] = pct(p_rdy);
      sim->m_axi_awready[b] = pct(p_rdy);
      sim->m_axi_wready[b] = pct(p_rdy);
      sim->m_axi_rvalid[b] = s.r_valid;
      sim->m_axi_rdata[b] = s.r_cur.data;
      sim->m_axi_rid[b] = s.r_cur.id;
      sim->m_axi_rlast[b] = 1;
      sim->m_axi_rresp[b] = 0;
      sim->m_axi_bvalid[b] = s.b_valid;
      sim->m_axi_bid[b] = s.b_cur.id;
      sim->m_axi_bresp[b] = 0;
    }

    sim->eval();

    bool progress = false;
    // master-side protocol: a presented request holds until its handshake
    for (int b = 0; b < NB; ++b) {
      uint64_t ar = (uint64_t(sim->m_axi_araddr[b]) << 32) | uint64_t(sim->m_axi_arid[b]);
      uint64_t aw = (uint64_t(sim->m_axi_awaddr[b]) << 32) | uint64_t(sim->m_axi_awid[b]);
      uint64_t w = sim->m_axi_wdata[b] ^ (uint64_t(sim->m_axi_wstrb[b]) << 56);
      if (p_ar[b] && !(sim->m_axi_arvalid[b] && ar == p_ar_v[b])) fail(c, "AR dropped or changed before arready", b);
      if (p_aw[b] && !(sim->m_axi_awvalid[b] && aw == p_aw_v[b])) fail(c, "AW dropped or changed before awready", b);
      if (p_w[b] && !(sim->m_axi_wvalid[b] && w == p_w_v[b])) fail(c, "W dropped or changed before wready", b);
      p_ar[b] = sim->m_axi_arvalid[b] && !sim->m_axi_arready[b]; p_ar_v[b] = ar;
      p_aw[b] = sim->m_axi_awvalid[b] && !sim->m_axi_awready[b]; p_aw_v[b] = aw;
      p_w[b]  = sim->m_axi_wvalid[b] && !sim->m_axi_wready[b];   p_w_v[b] = w;
      if (sim->m_axi_arvalid[b] && sim->m_axi_arlen[b] != 0) fail(c, "burst read", b);
    }

    // requester handshakes
    for (int p = 0; p < NI; ++p) {
      auto& r = req[p];
      if (r.valid && sim->mem_req_ready[p]) {
        progress = true;
        if (r.rw) {
          auto it = r.view.find(r.addr);
          uint64_t old = (it == r.view.end()) ? init_value(r.addr) : it->second;
          r.view[r.addr] = merge(old, r.data, r.byteen);
          ++n_wr;
        } else {
          auto it = r.view.find(r.addr);
          r.expect[r.tag] = (it == r.view.end()) ? init_value(r.addr) : it->second;
          r.free_tags.pop_front();
          ++n_rd;
        }
        r.valid = false;
      }
      if (sim->mem_rsp_valid[p] && sim->mem_rsp_ready[p]) {
        progress = true;
        uint32_t tag = sim->mem_rsp_tag[p];
        auto it = r.expect.find(tag);
        if (it == r.expect.end()) {
          fail(c, "response with a tag not outstanding on port", p);
        } else {
          if (it->second != uint64_t(sim->mem_rsp_data[p])) fail(c, "stale or wrong read data on port", p);
          r.expect.erase(it);
          r.free_tags.push_back(tag);
          ++n_rsp;
        }
      }
    }

    // slave handshakes: writes first become visible when their B is sent
    for (int b = 0; b < NB; ++b) {
      auto& s = slv[b];
      if (s.b_valid && sim->m_axi_bready[b]) {
        progress = true;
        mem[s.b_cur.addr] = merge(mem_get(s.b_cur.addr), s.b_cur.data, s.b_cur.strb);
        s.b_valid = false;
      }
      if (s.r_valid && sim->m_axi_rready[b]) {
        progress = true;
        s.r_valid = false;
      }
      if (sim->m_axi_arvalid[b] && sim->m_axi_arready[b]) {
        progress = true;
        uint32_t a = uint32_t(sim->m_axi_araddr[b]) >> 3;
        bool raw = false;
        for (auto& w : s.writes) raw |= (w.addr == a);
        for (auto& w : s.aws) raw |= (w.addr == a);
        if (s.b_valid && s.b_cur.addr == a) raw = true;
        if (raw) fail(c, "read issued while a write to its line is unacknowledged, bank", b);
        s.reads.push_back({uint32_t(sim->m_axi_arid[b]), mem_get(a), c + 1 + rng() % 64});
      }
      if (sim->m_axi_awvalid[b] && sim->m_axi_awready[b]) {
        progress = true;
        if (sim->m_axi_awlen[b] != 0) fail(c, "burst write", b);
        s.aws.push_back({uint32_t(sim->m_axi_awid[b]), uint32_t(sim->m_axi_awaddr[b]) >> 3});
      }
      if (sim->m_axi_wvalid[b] && sim->m_axi_wready[b]) {
        progress = true;
        s.ws.push_back({uint64_t(sim->m_axi_wdata[b]), uint32_t(sim->m_axi_wstrb[b])});
      }
      while (!s.aws.empty() && !s.ws.empty()) {
        s.writes.push_back({s.aws.front().id, s.aws.front().addr, s.ws.front().data, s.ws.front().strb, c + 1 + rng() % 96});
        s.aws.pop_front();
        s.ws.pop_front();
      }
    }

    bool pending = false;
    for (int p = 0; p < NI; ++p) pending |= req[p].valid || !req[p].expect.empty();
    for (int b = 0; b < NB; ++b)
      pending |= slv[b].r_valid || slv[b].b_valid || !slv[b].reads.empty() || !slv[b].writes.empty() || !slv[b].aws.empty() || !slv[b].ws.empty();
    if (progress) last_progress = c;
    if (pending && c - last_progress > WATCHDOG) {
      fail(c, "no progress: deadlock", 0);
      for (int p = 0; p < NI; ++p)
        printf("  port %d: req valid=%d rw=%d addr=%x ready=%d outstanding=%zu\n", p, req[p].valid, req[p].rw, req[p].addr,
               sim->mem_req_ready[p], req[p].expect.size());
      for (int b = 0; b < NB; ++b)
        printf("  bank %d: arvalid=%d awvalid=%d wvalid=%d reads=%zu writes=%zu aw=%zu w=%zu rvalid=%d bvalid=%d\n", b,
               sim->m_axi_arvalid[b], sim->m_axi_awvalid[b], sim->m_axi_wvalid[b], slv[b].reads.size(), slv[b].writes.size(),
               slv[b].aws.size(), slv[b].ws.size(), slv[b].r_valid, slv[b].b_valid);
    }
    if (!issue && !pending) break;

    ticks = sim.step(ticks, 2);
  }

  bool drained = true;
  for (int p = 0; p < NI; ++p) drained &= !req[p].valid && req[p].expect.empty();
  if (errors == 0 && !drained) fail(c, "not drained at the end", 0);
  printf("mem_to_axi NI=%d NB=%d TIDW=%d seed=%u: cycles=%lu reads=%lu writes=%lu responses=%lu errors=%lu\n%s\n",
         NI, NB, TIDW, seed, (unsigned long)c, (unsigned long)n_rd, (unsigned long)n_wr, (unsigned long)n_rsp,
         (unsigned long)errors, errors ? "FAILED" : "PASSED");
  return errors ? 1 : 0;
}
