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

// The properties under test, for the ID-routed (MULTI_OUT) 2:1 arbiter:
//  * the slave sees each write's data in the order it saw the addresses,
//    whole bursts at a time, whichever channel each master offers first;
//  * an offer the slave has not taken stays unchanged (AXI4 A3.2.1);
//  * no master waits on AWREADY for its data, so a slave that takes the
//    data before the address cannot deadlock it;
//  * B and R reach the master that issued them, in order;
//  * a write whose address and data are offered together moves in one cycle.

#include <cstdio>
#include <cstdint>
#include <deque>
#include <vector>
#include "VVX_mm_axi_arb_top.h"
#include "verilated.h"

namespace {

constexpr int NUM_MASTERS = 2;
constexpr int ID_SRC_BIT  = 7;    // the arbiter's source field (ID_WIDTH-1)

VVX_mm_axi_arb_top* dut = nullptr;
int failures = 0;
uint32_t rng = 1;

uint32_t rnd() {
  rng = rng * 1103515245u + 12345u;
  return rng >> 16;
}

bool chance(int pct) { return int(rnd() % 100) < pct; }

void check(bool ok, const char* what) {
  printf("  %-62s %s\n", what, ok ? "PASS" : "*** FAIL ***");
  if (!ok) {
    ++failures;
  }
}

int errors = 0;
void error(const char* msg, long a = 0, long b = 0) {
  if (errors < 10) {
    printf("    error: %s (%ld, %ld)\n", msg, a, b);
  }
  ++errors;
}

// Packed [N][W] port helpers.
template <typename T> uint32_t get(T v, int i, int w) { return uint32_t((uint64_t(v) >> (i * w)) & ((1ull << w) - 1)); }
template <typename T> void put(T& v, int i, int w, uint32_t x) {
  uint64_t m = ((1ull << w) - 1) << (i * w);
  v = T((uint64_t(v) & ~m) | ((uint64_t(x) << (i * w)) & m));
}

uint32_t waddr(int m, int k)          { return (uint32_t(m) << 28) | (uint32_t(k) << 8); }
uint32_t wdata(int m, int k, int beat) { return waddr(m, k) | uint32_t(beat); }

enum Style {
  STYLE_FREE,   // AW and W offered independently; data may lead its address
  STYLE_AWFIRST // data only after its address was taken, one write at a time
};

struct Master {
  int id;
  Style style;
  int pct;                  // offer probability per cycle
  int num_writes, num_reads;
  std::vector<int> len;     // per write, beats - 1
  int aw_next = 0, aw_done = 0;
  bool aw_offer = false;
  int w_next = 0, w_beat = 0, w_done = 0;
  bool w_offer = false;
  int b_recv = 0;
  int ar_next = 0;
  bool ar_offer = false;
  std::deque<std::pair<uint32_t, int>> r_exp;   // addr, len
  int r_beat = 0, r_done = 0;
  bool done() const { return b_recv == num_writes && r_done == num_reads; }
};

struct SlaveCfg {
  int aw_pct, w_pct, ar_pct, b_pct, r_pct;
  bool w_first;             // AWREADY only once that write's data has begun
};

struct AwRec { uint32_t addr; uint32_t id; int len; };

bool run(std::vector<Master>& ms, const SlaveCfg& sc, int max_cycles, int* cycles_out,
         int* writes_out = nullptr, int* last_write_cycle = nullptr) {
  std::deque<AwRec> aw_q;
  std::deque<std::vector<uint32_t>> w_bursts;
  std::vector<uint32_t> w_cur;
  int w_started = 0, aw_taken = 0;
  std::deque<uint32_t> b_q;
  std::deque<std::pair<uint32_t, int>> ar_q;  // addr|id, beats
  std::deque<AwRec> r_q;
  int r_beat = 0;
  bool b_hold = false, r_hold = false;
  int writes = 0;
  // offer-stability tracking on the slave side
  bool pend_aw = false, pend_w = false, pend_ar = false;
  uint64_t pend_aw_v = 0, pend_w_v = 0, pend_ar_v = 0;

  int cycle = 0;
  for (; cycle < max_cycles; ++cycle) {
    bool all_done = true;
    for (auto& m : ms) {
      all_done = all_done && m.done();
    }
    if (all_done && aw_q.empty() && w_bursts.empty() && b_q.empty() && r_q.empty()) {
      break;
    }

    // ---- masters drive ----
    dut->s_awvalid = 0; dut->s_wvalid = 0; dut->s_arvalid = 0;
    dut->s_wlast = 0; dut->s_bready = 0; dut->s_rready = 0;
    for (int i = 0; i < NUM_MASTERS; ++i) {
      auto& m = ms[i];
      if (!m.aw_offer && m.aw_next < m.num_writes) {
        bool ok = (m.style == STYLE_FREE) ? (m.aw_next - m.w_done <= 3)
                                          : (m.aw_next == m.w_done);
        m.aw_offer = ok && chance(m.pct);
      }
      if (m.aw_offer) {
        dut->s_awvalid |= (1 << i);
        put(dut->s_awaddr, i, 32, waddr(i, m.aw_next));
        put(dut->s_awid, i, 8, m.id);
        put(dut->s_awlen, i, 8, m.len[m.aw_next]);
      }
      if (!m.w_offer && m.w_next < m.num_writes) {
        bool ok = (m.style == STYLE_FREE) ? (m.w_next <= m.aw_done + 1)
                                          : (m.w_next < m.aw_done);
        m.w_offer = ok && (m.w_beat > 0 || chance(m.pct));
      }
      if (m.w_offer) {
        dut->s_wvalid |= (1 << i);
        put(dut->s_wdata, i, 32, wdata(i, m.w_next, m.w_beat));
        if (m.w_beat == m.len[m.w_next]) {
          dut->s_wlast |= (1 << i);
        }
      }
      if (!m.ar_offer && m.ar_next < m.num_reads) {
        m.ar_offer = chance(m.pct);
      }
      if (m.ar_offer) {
        dut->s_arvalid |= (1 << i);
        put(dut->s_araddr, i, 32, (uint32_t(i) << 28) | (uint32_t(m.ar_next) << 8) | 0x80);
        put(dut->s_arid, i, 8, m.id);
        put(dut->s_arlen, i, 8, m.ar_next % 3);
      }
      if (chance(70)) {
        dut->s_bready |= (1 << i);
      }
      if (chance(70)) {
        dut->s_rready |= (1 << i);
      }
    }

    // ---- slave drives ----
    dut->m_awready = chance(sc.aw_pct) && (!sc.w_first || w_started > aw_taken);
    dut->m_wready  = chance(sc.w_pct);
    dut->m_arready = chance(sc.ar_pct);
    if (!b_hold && !b_q.empty() && chance(sc.b_pct)) {
      b_hold = true;
    }
    dut->m_bvalid = b_hold;
    dut->m_bid    = b_hold ? b_q.front() : 0;
    if (!r_hold && !r_q.empty() && chance(sc.r_pct)) {
      r_hold = true;
    }
    dut->m_rvalid = r_hold;
    if (r_hold) {
      dut->m_rdata = r_q.front().addr | uint32_t(r_beat);
      dut->m_rid   = r_q.front().id;
      dut->m_rlast = (r_beat == r_q.front().len);
    }

    dut->clk = 0;
    dut->eval();

    // ---- offer stability, slave side ----
    uint64_t aw_v = (uint64_t(dut->m_awaddr) << 16) | (uint64_t(dut->m_awid) << 8) | dut->m_awlen;
    uint64_t w_v  = (uint64_t(dut->m_wdata) << 1) | dut->m_wlast;
    uint64_t ar_v = (uint64_t(dut->m_araddr) << 16) | (uint64_t(dut->m_arid) << 8) | dut->m_arlen;
    if (pend_aw && (!dut->m_awvalid || aw_v != pend_aw_v)) {
      error("AW offer changed before it was taken", cycle);
    }
    if (pend_w  && (!dut->m_wvalid  || w_v  != pend_w_v)) {
      error("W offer changed before it was taken", cycle);
    }
    if (pend_ar && (!dut->m_arvalid || ar_v != pend_ar_v)) {
      error("AR offer changed before it was taken", cycle);
    }
    pend_aw = dut->m_awvalid && !dut->m_awready; pend_aw_v = aw_v;
    pend_w  = dut->m_wvalid  && !dut->m_wready;  pend_w_v  = w_v;
    pend_ar = dut->m_arvalid && !dut->m_arready; pend_ar_v = ar_v;

    // ---- slave handshakes ----
    if (dut->m_awvalid && dut->m_awready) {
      aw_q.push_back({uint32_t(dut->m_awaddr), uint32_t(dut->m_awid), int(dut->m_awlen)});
      ++aw_taken;
    }
    if (dut->m_wvalid && dut->m_wready) {
      if (w_cur.empty()) {
        ++w_started;
      }
      w_cur.push_back(uint32_t(dut->m_wdata));
      if (dut->m_wlast) {
        w_bursts.push_back(w_cur);
        w_cur.clear();
      }
    }
    while (!aw_q.empty() && !w_bursts.empty()) {
      auto aw = aw_q.front(); auto& bu = w_bursts.front();
      if (int(bu.size()) != aw.len + 1) {
        error("burst length differs from its AW", bu.size(), aw.len);
      }
      for (size_t b = 0; b < bu.size(); ++b) {
        if (bu[b] != (aw.addr | uint32_t(b))) {
          error("W data does not belong to the AW at its position", bu[b], aw.addr);
        }
      }
      if (((aw.addr >> 28) & 1) != ((aw.id >> ID_SRC_BIT) & 1)) {
        error("AW id source field wrong", aw.addr, aw.id);
      }
      b_q.push_back(aw.id);
      aw_q.pop_front(); w_bursts.pop_front();
      ++writes;
      if (last_write_cycle) {
        *last_write_cycle = cycle;
      }
    }
    if (dut->m_arvalid && dut->m_arready) {
      r_q.push_back({uint32_t(dut->m_araddr), uint32_t(dut->m_arid), int(dut->m_arlen)});
    }
    bool b_fire = b_hold && dut->m_bready;
    bool r_fire = r_hold && dut->m_rready;

    // ---- master handshakes ----
    for (int i = 0; i < NUM_MASTERS; ++i) {
      auto& m = ms[i];
      if (m.aw_offer && get(dut->s_awready, i, 1)) {
        m.aw_offer = false;
        ++m.aw_next;
        ++m.aw_done;
      }
      if (m.w_offer && get(dut->s_wready, i, 1)) {
        m.w_offer = false;
        if (m.w_beat == m.len[m.w_next]) {
          m.w_beat = 0;
          ++m.w_next;
          ++m.w_done;
        } else {
          ++m.w_beat;
        }
      }
      if (m.ar_offer && get(dut->s_arready, i, 1)) {
        m.ar_offer = false;
        m.r_exp.push_back({(uint32_t(i) << 28) | (uint32_t(m.ar_next) << 8) | 0x80, m.ar_next % 3});
        ++m.ar_next;
      }
      if (get(dut->s_bvalid, i, 1) && get(dut->s_bready, i, 1)) {
        if (get(dut->s_bid, i, 8) != uint32_t(m.id)) {
          error("B to the wrong master or id", i, get(dut->s_bid, i, 8));
        }
        if (m.b_recv >= m.aw_done) {
          error("B before its write was issued", i, m.b_recv);
        }
        ++m.b_recv;
      }
      if (get(dut->s_rvalid, i, 1) && get(dut->s_rready, i, 1)) {
        if (m.r_exp.empty()) {
          error("unexpected R", i);
        } else {
          auto e = m.r_exp.front();
          if (get(dut->s_rid, i, 8) != uint32_t(m.id)) {
            error("R to the wrong master or id", i, get(dut->s_rid, i, 8));
          }
          if (get(dut->s_rdata, i, 32) != (e.first | uint32_t(m.r_beat))) {
            error("R data wrong", i, get(dut->s_rdata, i, 32));
          }
          bool last = (m.r_beat == e.second);
          if (get(dut->s_rlast, i, 1) != uint32_t(last)) {
            error("RLAST wrong", i, m.r_beat);
          }
          if (last) {
            m.r_exp.pop_front();
            m.r_beat = 0;
            ++m.r_done;
          } else {
            ++m.r_beat;
          }
        }
      }
    }

    dut->clk = 1;
    dut->eval();

    if (b_fire) {
      b_q.pop_front();
      b_hold = false;
    }
    if (r_fire) {
      if (r_beat == r_q.front().len) {
        r_q.pop_front();
        r_beat = 0;
      } else {
        ++r_beat;
      }
      r_hold = false;
    }
  }
  *cycles_out = cycle;
  if (writes_out) {
    *writes_out = writes;
  }
  for (auto& m : ms) {
    if (!m.done()) {
      return false;
    }
  }
  return true;
}

void do_reset() {
  dut->reset = 1;
  for (int i = 0; i < 4; ++i) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
  }
  dut->reset = 0;
  dut->clk = 0; dut->eval();
}

Master make_master(int id, Style style, int pct, int writes, int reads, int max_len) {
  Master m;
  m.id = id; m.style = style; m.pct = pct;
  m.num_writes = writes; m.num_reads = reads;
  for (int k = 0; k < writes; ++k) {
    m.len.push_back(int(rnd() % (max_len + 1)));
  }
  return m;
}

} // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new VVX_mm_axi_arb_top;
  int cycles = 0;

  printf("1. two masters, random traffic and back-pressure\n");
  for (uint32_t seed = 1; seed <= 8; ++seed) {
    rng = seed;
    errors = 0;
    do_reset();
    std::vector<Master> ms = { make_master(5, STYLE_FREE, 60, 400, 300, 3),
                               make_master(6, STYLE_AWFIRST, 60, 400, 300, 3) };
    SlaveCfg sc = {50, 50, 50, 60, 60, false};
    bool ok = run(ms, sc, 200000, &cycles);
    char what[96];
    snprintf(what, sizeof(what), "seed %u: all writes and reads complete in order (%d cycles)", seed, cycles);
    check(ok && errors == 0, what);
  }

  printf("2. a slave that takes the data before the address\n");
  for (uint32_t seed = 11; seed <= 14; ++seed) {
    rng = seed;
    errors = 0;
    do_reset();
    std::vector<Master> ms = { make_master(5, STYLE_FREE, 70, 400, 100, 3),
                               make_master(6, STYLE_FREE, 70, 400, 100, 3) };
    SlaveCfg sc = {60, 60, 60, 70, 70, true};
    bool ok = run(ms, sc, 200000, &cycles);
    char what[96];
    snprintf(what, sizeof(what), "seed %u: no deadlock, data still in address order", seed);
    check(ok && errors == 0, what);
  }

  printf("3. single-beat writes with address and data offered together\n");
  {
    rng = 21;
    errors = 0;
    do_reset();
    std::vector<Master> ms = { make_master(5, STYLE_FREE, 100, 1000, 0, 0),
                               make_master(6, STYLE_FREE, 0, 0, 0, 0) };
    SlaveCfg sc = {100, 100, 100, 100, 100, false};
    int writes = 0, last = 0;
    bool ok = run(ms, sc, 20000, &cycles, &writes, &last);
    char what[96];
    snprintf(what, sizeof(what), "one write per cycle (%d writes in %d cycles)", writes, last + 1);
    check(ok && errors == 0 && last + 1 <= writes + 16, what);
  }

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
         failures, failures == 1 ? "" : "s");
  delete dut;
  return failures ? 1 : 0;
}
