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

// Cluster TLB test: the sockets' miss requests through the L2 TLB and the
// device page-table walker, against an Sv39 page table held by a memory model
// with random latency and back-pressure. The working set overflows both the
// set-associative array (base pages crowded into few sets) and the megapage
// array (more superpages than entries), and includes every structural fault
// the walker raises. Flushes remap pages. Every request must be answered
// exactly once with the translation (or fault) of a mapping valid between its
// accept and its answer.

#include "vl_simulator.h"
#include "VVX_tlb_l2_top.h"
#include <iostream>
#include <deque>
#include <map>
#include <random>
#include <vector>
#include "VX_config.h"
#include "VX_types.h"

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

double sc_time_stamp() {
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

static constexpr int NS       = 4;    // VX_tlb_l2_top default
static constexpr int LVL_BITS = 9;
static constexpr int VPN_W    = 3 * LVL_BITS;
static constexpr int PPN_W    = VX_CFG_MEM_ADDR_WIDTH - 12;
static constexpr int LEVEL_W  = 2;
static constexpr int FLAGS_W  = 8;
static constexpr int DS       = VX_CFG_L1_LINE_SIZE;
static constexpr int NPTE     = DS / 8;

template <typename T>
static uint64_t bits(const T& v, int lo, int w) {
  return (uint64_t(v) >> lo) & ((w >= 64) ? ~0ull : ((1ull << w) - 1));
}
template <std::size_t N>
static uint64_t bits(const VlWide<N>& v, int lo, int w) {
  uint64_t r = 0;
  for (int i = 0; i < w; ++i) {
    int b = lo + i;
    r |= uint64_t((v[b / 32] >> (b % 32)) & 1) << i;
  }
  return r;
}
template <typename T>
static void set_bits(T& v, int lo, int w, uint64_t x) {
  for (int i = 0; i < w; ++i) {
    int b = lo + i;
    if ((x >> i) & 1) v |= T(1) << b; else v &= ~(T(1) << b);
  }
}
template <std::size_t N>
static void set_bits(VlWide<N>& v, int lo, int w, uint64_t x) {
  for (int i = 0; i < w; ++i) {
    int b = lo + i;
    if ((x >> i) & 1) v[b / 32] |= (1u << (b % 32)); else v[b / 32] &= ~(1u << (b % 32));
  }
}

// ---------------------------------------------------------------------------
// The page table. Root at ROOT; root slot 1 is a gigapage, 2 a misaligned
// gigapage, 3 invalid, others point to a level-1 table. In a level-1 table,
// j % 16 == 5 is a megapage (j % 32 == 5 moves on every flush), 13 a
// misaligned megapage, 9 invalid. In a leaf table, k % 11 == 5 is invalid,
// k % 13 == 7 is W without R, k % 17 == 3 a pointer at the last level, and
// k % 4 == 1 moves on every flush.
// ---------------------------------------------------------------------------
static constexpr uint32_t F_V = 1, F_R = 2, F_W = 4, F_X = 8, F_U = 16, F_A = 64, F_D = 128;
static constexpr uint64_t ROOT   = 0x100;
static constexpr uint64_t L1BASE = 0x1000;
static constexpr uint64_t L0BASE = 0x10000;

static uint64_t make_pte(uint64_t ppn, uint32_t flags) {
  return (ppn << 10) | flags;
}

static uint64_t pte_at(uint64_t table, uint64_t idx, int gen) {
  const uint32_t leaf = F_V | F_R | F_W | F_X | F_U | F_A | F_D;
  if (table == ROOT) {
    if (idx == 1) return make_pte(0x40000ull, leaf);
    if (idx == 2) return make_pte(0x40000ull | 0x200, leaf);
    if (idx == 3) return 0;
    return make_pte(L1BASE + idx, F_V);
  }
  if (table >= L1BASE && table < L1BASE + 512) {
    uint64_t i = table - L1BASE;
    if (idx % 16 == 5) {
      uint64_t g = (idx % 32 == 5) ? uint64_t(gen & 3) : 0;
      return make_pte((0x300ull + ((i * 512 + idx) * 4 + g)) << LVL_BITS, leaf & ~((idx % 3 == 0) ? F_X : 0));
    }
    if (idx % 16 == 13) return make_pte((0x300ull << LVL_BITS) | 0x5, leaf);
    if (idx % 16 == 9) return 0;
    return make_pte(L0BASE + i * 512 + idx, F_V);
  }
  if (table >= L0BASE && table < L0BASE + 512 * 512) {
    uint64_t t = table - L0BASE;
    if (idx % 11 == 5) return 0;
    if (idx % 13 == 7) return make_pte(0x1234, F_V | F_W | F_U);
    if (idx % 17 == 3) return make_pte(L0BASE, F_V);
    uint64_t g = (idx % 4 == 1) ? uint64_t(gen) : 0;
    uint64_t ppn = (((t << LVL_BITS) | idx) * 0x9e37ull ^ (g << 20) ^ 0x5a5a5ull) & ((1ull << PPN_W) - 1);
    return make_pte(ppn, leaf & ~((idx % 7 == 2) ? F_W : 0));
  }
  return 0;
}

struct Walk { bool fault; int level; uint64_t ppn; uint32_t flags; };

// Sv39 walk of the table above, with the walker's structural faults.
static Walk walk(uint64_t vpn, int gen) {
  uint64_t table = ROOT;
  for (int level = 2; level >= 0; --level) {
    uint64_t idx = (vpn >> (level * LVL_BITS)) & 511;
    uint64_t pte = pte_at(table, idx, gen);
    uint32_t flags = uint32_t(pte & 0xff);
    uint64_t ppn = (pte >> 10) & ((1ull << PPN_W) - 1);
    bool v = flags & F_V, r = flags & F_R, w = flags & F_W, x = flags & F_X;
    bool leaf = r || w || x;
    if (!v || (!r && w)) return {true, 0, 0, 0};
    if (leaf) {
      uint64_t sp_mask = (1ull << (level * LVL_BITS)) - 1;
      if (ppn & sp_mask) return {true, 0, 0, 0};
      return {false, level, ppn, flags};
    }
    if (level == 0) return {true, 0, 0, 0};
    table = ppn;
  }
  return {true, 0, 0, 0};
}

static uint64_t vpn_of(uint64_t v2, uint64_t v1, uint64_t v0) {
  return (v2 << (2 * LVL_BITS)) | (v1 << LVL_BITS) | v0;
}

struct Req {
  int socket; uint32_t id; uint64_t vpn; int acc; bool amo; int gen_in;
  int accept_cycle = -1;
  bool done = false;
};

struct MemRsp {
  std::remove_reference_t<decltype(VVX_tlb_l2_top::mem_req_tag)> tag;
  uint64_t addr;
  int due;
};

static int errors = 0;
static int cycle = 0;
#define CHECK(cond, msg) do { if (!(cond)) { if (errors < 20) std::cout << "cycle " << cycle << ": " << msg << std::endl; ++errors; } } while (0)

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  unsigned seed = 1;
  int num_reqs = 30000;
  // --fair: every socket always presents a miss to a fresh, validly mapped base
  // page, answers are always taken and memory answers at a fixed latency, so
  // the walker stays saturated; every request must be answered within
  // FAIR_BOUND cycles of its accept (a starved miss entry fails the test).
  bool fair = false;
  static constexpr int FAIR_BOUND = 3000;
  int max_wait = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--seed=", 0) == 0) seed = std::stoul(a.substr(7));
    if (a.rfind("--reqs=", 0) == 0) num_reqs = std::stoi(a.substr(7));
    if (a == "--fair") fair = true;
  }
  std::mt19937 rng(seed);

  // The working set: base pages crowded into 4 sets of the main array (and
  // spread across many leaf tables), 24 megapages (3 times the megapage
  // array), the gigapage, and every fault kind. Keep the class order: the
  // request mix below indexes it.
  std::vector<uint64_t> pages;
  for (int i = 0; i < 48; ++i) {
    uint64_t v0 = uint64_t((i % 4) * 32) | (uint64_t((i / 4) % 4) << 7);   // VPN[6:0] picks the set
    pages.push_back(vpn_of(4 + i % 3, 2 + (i / 12) * 4, v0));
  }
  for (int i = 0; i < 24; ++i) pages.push_back(vpn_of(4 + i % 5, 5 + 16 * (i / 5), rng() % 512));
  pages.push_back(vpn_of(1, rng() % 512, rng() % 512));       // gigapage
  pages.push_back(vpn_of(1, rng() % 512, rng() % 512));
  pages.push_back(vpn_of(2, 7, 9));                           // misaligned gigapage
  pages.push_back(vpn_of(3, 7, 9));                           // invalid root slot
  pages.push_back(vpn_of(4, 13, 9));                          // misaligned megapage
  pages.push_back(vpn_of(4, 9, 9));                           // invalid level-1 slot
  pages.push_back(vpn_of(4, 2, 5));                           // invalid leaf
  pages.push_back(vpn_of(4, 2, 7));                           // W without R
  pages.push_back(vpn_of(4, 2, 3));                           // pointer at the last level

  vl_simulator<VVX_tlb_l2_top> sim;
  uint64_t tick = 0;
  sim->satp = (8ull << 60) | ROOT;
  sim->req_valid = 0;
  sim->rsp_ready = 0;
  sim->mem_req_ready = 0;
  sim->mem_rsp_valid = 0;
  sim->flush_req = 0;
  tick = sim.reset(tick);

  const int IDW = sim->id_width;
  const uint32_t NUM_IDS = 1u << IDW;
  std::vector<Req> reqs;
  std::vector<int> pending(NS, -1);
  std::vector<std::map<uint32_t, int>> outstanding(NS);   // id -> request
  std::vector<MemRsp> mem_q;
  int mem_sel = -1;
  int gen = 0, flushes = 0, issued = 0, answered = 0, faults = 0, mem_reads = 0, fault_reports = 0;
  bool flush_hold = false;
  // fresh base pages with a leaf mapping: level-1 slots and leaf indices that
  // the table above maps to an ordinary page
  uint64_t fair_v1 = 0, fair_v0 = 0;
  auto next_fair_vpn = [&]() {
    for (;;) {
      if (++fair_v0 == 512) {
        fair_v0 = 0;
        do { ++fair_v1; } while (fair_v1 % 16 == 5 || fair_v1 % 16 == 13 || fair_v1 % 16 == 9);
      }
      uint64_t k = fair_v0;
      if (k % 11 == 5 || k % 13 == 7 || k % 17 == 3) continue;
      return vpn_of(4, fair_v1, k);
    }
  };

  for (cycle = 0; cycle < 4000000; ++cycle) {
    // Alternate phases: a mixed working set at a long memory latency, then a
    // megapage storm with fast walks and slow answers, which holds lookups in
    // stage R while fills replace the megapage entries they matched.
    bool storm = ((issued / 2500) % 2) == 1;

    // requests
    for (int s = 0; s < NS; ++s) {
      if (pending[s] < 0 && issued < num_reqs && !flush_hold && outstanding[s].size() < NUM_IDS
       && (fair || rng() % 3 != 0)) {
        uint32_t id = rng() % NUM_IDS;
        while (outstanding[s].count(id)) id = (id + 1) % NUM_IDS;
        Req r;
        r.socket = s;
        r.id = id;
        // 40% base pages (5% in a storm), the rest megapages and 15% the
        // gigapage and the faults; superpage requests vary the VPN bits below
        // their level
        uint32_t c = rng() % 20;
        if (fair) {
          r.vpn = next_fair_vpn();
        } else if (c < (storm ? 1u : 8u)) {
          r.vpn = pages[rng() % 48];
        } else if (c < 17) {
          r.vpn = (pages[48 + rng() % 24] & ~511ull) | (rng() % 512);
        } else {
          size_t k = 72 + rng() % (pages.size() - 72);
          r.vpn = (k < 74) ? ((pages[k] & ~0x3ffffull) | (rng() % 0x40000)) : pages[k];
        }
        r.acc = fair ? 0 : int(rng() % 3);
        r.amo = !fair && (rng() % 8 == 0);
        r.gen_in = -1;
        pending[s] = int(reqs.size());
        outstanding[s][id] = pending[s];
        reqs.push_back(r);
        ++issued;
      }
      int q = pending[s];
      set_bits(sim->req_valid, s, 1, q >= 0);
      if (q >= 0) {
        const Req& r = reqs[q];
        set_bits(sim->req_id, s * IDW, IDW, r.id);
        set_bits(sim->req_vpn, s * VPN_W, VPN_W, r.vpn);
        set_bits(sim->req_access, s * 2, 2, uint64_t(r.acc));
        set_bits(sim->req_amo, s, 1, r.amo);
      }
    }
    sim->rsp_ready = 0;
    for (int s = 0; s < NS; ++s) set_bits(sim->rsp_ready, s, 1, fair || (rng() % 8) < (storm ? 1u : 4u));
    sim->mem_req_ready = fair || (rng() % 4 != 0);
    if (!fair && !flush_hold && issued > 200 && issued < num_reqs && rng() % 4000 == 0) flush_hold = true;
    sim->flush_req = flush_hold;

    // memory: one response at a time, any due request, in random order
    if (mem_sel < 0) {
      std::vector<int> due;
      for (int i = 0; i < int(mem_q.size()); ++i) {
        if (mem_q[i].due <= cycle) due.push_back(i);
      }
      if (!due.empty()) {
        mem_sel = due[rng() % due.size()];
        const MemRsp& m = mem_q[mem_sel];
        uint64_t byte = m.addr * DS;
        uint64_t table = byte >> 12;
        uint64_t idx0 = (byte & 0xfff) >> 3;
        for (int i = 0; i < NPTE; ++i) {
          set_bits(sim->mem_rsp_data, i * 64, 64, pte_at(table, idx0 + i, gen));
        }
        sim->mem_rsp_tag = m.tag;
      }
    }
    sim->mem_rsp_valid = (mem_sel >= 0);
    sim->eval();

    // sample this cycle's handshakes
    std::vector<int> acc;
    for (int s = 0; s < NS; ++s) {
      if (pending[s] >= 0 && bits(sim->req_ready, s, 1)) acc.push_back(s);
    }
    for (int s = 0; s < NS; ++s) {
      if (!(bits(sim->rsp_valid, s, 1) && bits(sim->rsp_ready, s, 1))) continue;
      uint32_t id = uint32_t(bits(sim->rsp_id, s * IDW, IDW));
      auto it = outstanding[s].find(id);
      CHECK(it != outstanding[s].end(), "socket " << s << " answer for idle id " << id);
      if (it == outstanding[s].end()) continue;
      Req& r = reqs[it->second];
      CHECK(r.gen_in >= 0, "request " << it->second << " answered before its accept");
      bool fault = bits(sim->rsp_fault, s, 1);
      int level = int(bits(sim->rsp_level, s * LEVEL_W, LEVEL_W));
      uint64_t ppn = bits(sim->rsp_ppn, s * PPN_W, PPN_W);
      uint32_t flags = uint32_t(bits(sim->rsp_flags, s * FLAGS_W, FLAGS_W));
      bool ok = false;
      for (int g = std::max(r.gen_in, 0); g <= gen && !ok; ++g) {
        Walk w = walk(r.vpn, g);
        ok = w.fault ? fault : (!fault && level == w.level && ppn == w.ppn && flags == w.flags);
      }
      if (!ok) {
        Walk w = walk(r.vpn, gen);
        CHECK(ok, "request " << it->second << " vpn " << std::hex << r.vpn << ": fault " << fault << " level " << level
              << " ppn " << ppn << " flags " << flags << ", expected fault " << w.fault << " level " << w.level
              << " ppn " << w.ppn << " flags " << w.flags << std::dec);
      }
      faults += fault;
      if (fair) {
        int wait = cycle - r.accept_cycle;
        max_wait = std::max(max_wait, wait);
        CHECK(wait <= FAIR_BOUND, "request " << it->second << " (socket " << s << ") answered " << wait
              << " cycles after its accept");
      }
      r.done = true;
      ++answered;
      outstanding[s].erase(it);
    }
    bool mreq_fire = sim->mem_req_valid && sim->mem_req_ready;
    MemRsp nm{};
    if (mreq_fire) {
      CHECK(!sim->mem_req_rw, "walker write");
      nm.tag = sim->mem_req_tag;
      nm.addr = sim->mem_req_addr;
      nm.due = cycle + (fair ? 30 : 2 + int(rng() % (storm ? 4 : 40)));
    }
    bool mrsp_fire = sim->mem_rsp_valid && sim->mem_rsp_ready;
    bool fdone = sim->flush_req && sim->flush_done;
    fault_reports += sim->fault_valid;

    tick = sim.step(tick, 2);
    ++timestamp;

    for (int s : acc) {
      reqs[pending[s]].gen_in = gen;
      reqs[pending[s]].accept_cycle = cycle;
      pending[s] = -1;
    }
    if (mrsp_fire) { mem_q.erase(mem_q.begin() + mem_sel); mem_sel = -1; }
    if (mreq_fire) { mem_q.push_back(nm); ++mem_reads; }
    if (fdone) { flush_hold = false; ++gen; ++flushes; }

    if (fair && (cycle % 1000) == 0) {
      bool starved = false;
      for (int s = 0; s < NS && !starved; ++s) {
        for (auto& [id, q] : outstanding[s]) {
          const Req& r = reqs[q];
          if (r.accept_cycle >= 0 && cycle - r.accept_cycle > FAIR_BOUND) {
            CHECK(false, "request " << q << " (socket " << s << ", id " << id << ") waiting "
                  << (cycle - r.accept_cycle) << " cycles since its accept");
            starved = true;
            break;
          }
        }
      }
      if (starved) break;
    }

    if (issued == num_reqs) {
      bool all = true;
      for (int s = 0; s < NS; ++s) all &= (pending[s] < 0) && outstanding[s].empty();
      if (all && sim->empty && mem_q.empty() && !flush_hold) break;
    }
  }

  int lost = 0;
  for (size_t i = 0; i < reqs.size(); ++i) {
    const Req& r = reqs[i];
    if (r.done) continue;
    if (fair && lost < 10 && r.accept_cycle >= 0) std::cout << "  (accepted at cycle " << r.accept_cycle << ")" << std::endl;
    if (++lost <= 10) std::cout << "request " << i << " never answered (socket " << r.socket << ", id " << r.id
                                << ", vpn " << std::hex << r.vpn << std::dec << ", gen " << r.gen_in << ")" << std::endl;
  }
  CHECK(lost == 0, lost << " requests lost");
  CHECK(issued == num_reqs, "only " << issued << " requests issued");

  std::cout << "seed " << seed << ": " << issued << " requests, " << answered << " answered, " << faults << " faults, "
            << mem_reads << " PTE reads, " << fault_reports << " fault reports, " << flushes << " flushes, "
            << cycle << " cycles, " << errors << " errors";
  if (fair) std::cout << ", max wait " << max_wait;
  std::cout << std::endl;
  if (errors) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
