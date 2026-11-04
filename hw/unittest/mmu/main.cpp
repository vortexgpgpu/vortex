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

// MMU test: random requests on every lane over more pages than the TLB holds
// (base pages, superpages, read-only pages, unmapped pages, bypassed IO),
// a walker model answering misses after random delays, random back-pressure,
// and flushes that remap pages. Every request is checked to leave exactly
// once: translated to a mapping valid between its accept and its departure,
// or bypassed unchanged, or (a faulting access) torn down with a zeroed
// response for a load / atomic and silently for a store. Requests of one lane
// to one page leave in order.

#include "vl_simulator.h"
#include "VVX_mmu_top.h"
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

static constexpr int NL       = 4;    // VX_mmu_top defaults
static constexpr int ADDR_W   = VX_CFG_MEM_ADDR_WIDTH - 3;
#if !defined(NDEBUG) || defined(SCOPE)
static constexpr int TAG_W    = 44 + 16;  // UUID_WIDTH + 16
#else
static constexpr int TAG_W    = 1 + 16;
#endif
static constexpr int OFF_W    = VX_VM_PAGE_LOG2_SIZE - 3;
static constexpr int LVL_BITS = 9;
static constexpr int PPN_W    = VX_CFG_MEM_ADDR_WIDTH - VX_VM_PAGE_LOG2_SIZE;

// bit field of a Verilator signal, scalar or wide
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
template <typename T>
static void clear_all(T& v) { v = 0; }
template <std::size_t N>
static void clear_all(VlWide<N>& v) { for (std::size_t i = 0; i < N; ++i) v[i] = 0; }

// The page table: a pool of base pages and one 2 MB superpage region.
struct Pte { bool fault; int level; uint64_t ppn; uint32_t flags; };
static constexpr uint32_t F_V = 1, F_R = 2, F_W = 4, F_X = 8, F_U = 16, F_A = 64, F_D = 128;
static constexpr uint64_t SP_VPN = 0x3a5ull << LVL_BITS;   // superpage base VPN (level 1)

static Pte walk(uint64_t vpn, int gen) {
  Pte p;
  p.fault = false; p.level = 0;
  p.flags = F_V | F_R | F_W | F_X | F_U | F_A | F_D;
  if ((vpn >> LVL_BITS) == (SP_VPN >> LVL_BITS)) {
    p.level = 1;
    p.ppn = (0x123ull + gen) << LVL_BITS;
    return p;
  }
  int k = int(vpn & 0x3f);
  if (k % 11 == 5) { p.fault = true; p.ppn = 0; p.flags = 0; return p; }   // unmapped
  if (k % 7 == 3) p.flags &= ~F_W;                                       // read-only
  // pages k % 4 == 1 move on every flush
  uint64_t g = (k % 4 == 1) ? uint64_t(gen) : 0;
  p.ppn = ((vpn * 0x9e37ull) ^ (g << 20) ^ 0x5a5a5ull) & ((1ull << PPN_W) - 1);
  return p;
}

static uint64_t translate(const Pte& p, uint64_t vpn) {
  uint64_t low = (p.level == 1) ? (vpn & ((1ull << LVL_BITS) - 1)) : 0;
  return p.ppn | low;
}

struct Req {
  int lane; bool rw, amo, io; uint64_t addr; int gen_in; int seq;
  bool done = false;
};

static int errors = 0;
static int cycle = 0;
#define CHECK(cond, msg) do { if (!(cond)) { if (errors < 20) std::cout << "cycle " << cycle << ": " << msg << std::endl; ++errors; } } while (0)

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  unsigned seed = 1;
  int num_reqs = 20000;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--seed=", 0) == 0) seed = std::stoul(a.substr(7));
    if (a.rfind("--reqs=", 0) == 0) num_reqs = std::stoi(a.substr(7));
  }
  std::mt19937 rng(seed);

  // 24 base pages (more than the TLB holds) and the superpage region
  std::vector<uint64_t> pages;
  for (int i = 0; i < 24; ++i) pages.push_back(0x4000ull + i * 5 + (i % 3) * 0x40);
  for (int i = 0; i < 4; ++i) pages.push_back(SP_VPN + rng() % (1u << LVL_BITS));

  vl_simulator<VVX_mmu_top> sim;
  uint64_t tick = 0;
  sim->vm_active = 1;
  sim->in_valid = 0;
  sim->walk_ready = 0;
  sim->fill_valid = 0;
  sim->flush_req = 0;
  sim->out_ready = 0;
  sim->crsp_ready = 0;
  tick = sim.reset(tick);

  std::vector<Req> reqs;
  std::vector<int> pending(NL, -1);                   // request presented on each lane
  std::map<std::pair<int, uint64_t>, int> last_seq;  // (lane, vpn) -> last seq out
  struct Walk { int id; uint64_t vpn; int due; };
  std::deque<Walk> walks;
  int gen = 0, flushes = 0, issued = 0, outs = 0, kills = 0, faults = 0, walks_n = 0;
  bool flush_hold = false;
  std::vector<int> seq_ctr(NL, 0);

  auto expect_fault = [&](const Req& r, const Pte& p) {
    return p.fault || ((r.rw || r.amo) && !(p.flags & F_W));
  };

  for (cycle = 0; cycle < 4000000; ++cycle) {
    // drive the inputs
    for (int l = 0; l < NL; ++l) {
      if (pending[l] < 0 && issued < num_reqs && !flush_hold && (rng() % 3 != 0)) {
        Req r;
        r.lane = l;
        uint64_t vpn = pages[rng() % pages.size()];
        r.io  = (rng() % 10 == 0);
        r.rw  = (rng() % 3 == 0);
        r.amo = !r.rw && (rng() % 12 == 0);
        r.addr = (vpn << OFF_W) | (rng() % (1u << OFF_W));
        r.seq = seq_ctr[l]++;
        r.gen_in = -1;
        pending[l] = int(reqs.size());
        reqs.push_back(r);
        ++issued;
      }
      int id = pending[l];
      set_bits(sim->in_valid, l, 1, id >= 0);
      if (id >= 0) {
        const Req& r = reqs[id];
        set_bits(sim->in_rw, l, 1, r.rw);
        set_bits(sim->in_io, l, 1, r.io);
        set_bits(sim->in_amo, l, 1, r.amo);
        set_bits(sim->in_addr, l * ADDR_W, ADDR_W, r.addr);
        set_bits(sim->in_tag, l * TAG_W, TAG_W, uint64_t(id));
      }
    }
    sim->out_ready  = rng() % 16;
    sim->crsp_ready = rng() % 16;
    sim->walk_ready = (rng() % 4 != 0);
    if (!flush_hold && issued > 100 && issued < num_reqs && rng() % 300 == 0) flush_hold = true;
    sim->flush_req = flush_hold;
    if (!sim->fill_valid && !walks.empty() && walks.front().due <= cycle) {
      const Walk& w = walks.front();
      Pte p = walk(w.vpn, gen);
      sim->fill_valid = 1;
      sim->fill_id    = w.id;
      sim->fill_fault = p.fault;
      sim->fill_level = p.level;
      sim->fill_ppn   = p.ppn;
      sim->fill_flags = p.flags;
    }
    sim->eval();

    // sample the handshakes of this cycle
    std::vector<int> acc;
    for (int l = 0; l < NL; ++l) {
      if (pending[l] >= 0 && bits(sim->in_ready, l, 1)) acc.push_back(l);
    }
    std::vector<std::pair<int, uint64_t>> outv;   // (lane, tag)
    for (int l = 0; l < NL; ++l) {
      if (bits(sim->out_valid, l, 1) && bits(sim->out_ready, l, 1)) {
        uint64_t tag = bits(sim->out_tag, l * TAG_W, TAG_W);
        CHECK(tag < reqs.size(), "unknown tag " << tag);
        if (tag >= reqs.size()) continue;
        Req& r = reqs[tag];
        CHECK(!r.done, "request " << tag << " left twice");
        CHECK(r.gen_in >= 0, "request " << tag << " left before its accept");
        CHECK(r.lane == l, "request " << tag << " left on lane " << l);
        CHECK(bool(bits(sim->out_rw, l, 1)) == r.rw && bool(bits(sim->out_io, l, 1)) == r.io
              && bool(bits(sim->out_amo, l, 1)) == r.amo, "request " << tag << " fields");
        uint64_t oaddr = bits(sim->out_addr, l * ADDR_W, ADDR_W);
        uint64_t vpn = r.addr >> OFF_W;
        if (r.io) {
          CHECK(oaddr == r.addr, "bypass " << tag << " translated");
        } else {
          bool ok = false;
          for (int g = r.gen_in; g <= gen && !ok; ++g) {
            Pte p = walk(vpn, g);
            if (!expect_fault(r, p)
             && oaddr == ((translate(p, vpn) << OFF_W) | (r.addr & ((1u << OFF_W) - 1)))) ok = true;
          }
          CHECK(ok, "request " << tag << " vpn " << std::hex << vpn << " -> " << (oaddr >> OFF_W) << std::dec << " wrong");
          auto key = std::make_pair(l, vpn);
          auto it = last_seq.find(key);
          CHECK(it == last_seq.end() || it->second < r.seq, "lane " << l << " page order");
          last_seq[key] = r.seq;
        }
        r.done = true;
        ++outs;
      }
      if (bits(sim->crsp_valid, l, 1) && bits(sim->crsp_ready, l, 1)) {
        uint64_t tag = bits(sim->crsp_tag, l * TAG_W, TAG_W);
        CHECK(tag < reqs.size(), "unknown teardown tag " << tag);
        if (tag >= reqs.size()) continue;
        Req& r = reqs[tag];
        CHECK(!r.done, "request " << tag << " torn down after leaving");
        CHECK(!r.io && (!r.rw || r.amo), "teardown of request " << tag << " that owes none");
        CHECK(bits(sim->crsp_data, l * 64, 64) == 0, "teardown data");
        bool may = false;
        for (int g = std::max(r.gen_in, 0); g <= gen; ++g) may |= expect_fault(r, walk(r.addr >> OFF_W, g));
        CHECK(may, "request " << tag << " torn down on a good page");
        r.done = true;
        ++kills;
      }
    }
    bool walk_fire = sim->walk_valid && sim->walk_ready;
    Walk nw{0, 0, 0};
    if (walk_fire) { nw.id = sim->walk_id; nw.vpn = sim->walk_vpn; nw.due = cycle + 2 + int(rng() % 30); }
    bool fill_fire = sim->fill_valid && sim->fill_ready;
    bool fdone = sim->flush_req && sim->flush_done;
    faults += sim->fault_valid;

    tick = sim.step(tick, 2);
    ++timestamp;

    for (int l : acc) {
      reqs[pending[l]].gen_in = gen;
      pending[l] = -1;
    }
    if (walk_fire) { walks.push_back(nw); ++walks_n; }
    if (fill_fire) { walks.pop_front(); sim->fill_valid = 0; }
    if (fdone) { flush_hold = false; ++gen; ++flushes; }

    if (issued == num_reqs) {
      bool all = true;
      for (int l = 0; l < NL; ++l) all &= (pending[l] < 0);
      if (all && sim->empty && walks.empty() && !sim->fill_valid && !flush_hold) {
        // the MMU is empty: every request has left or was torn down
        break;
      }
    }
  }

  // stores to faulting pages leave no trace: account for them
  int silent = 0, lost = 0;
  for (size_t i = 0; i < reqs.size(); ++i) {
    const Req& r = reqs[i];
    if (r.done) continue;
    bool may = false;
    for (int g = std::max(r.gen_in, 0); g <= gen; ++g) may |= expect_fault(r, walk(r.addr >> OFF_W, g));
    if (r.rw && !r.amo && !r.io && may) { ++silent; continue; }
    if (++lost <= 10) std::cout << "request " << i << " never left (lane " << r.lane << ", vpn " << std::hex << (r.addr >> OFF_W) << std::dec << ", rw " << r.rw << ", amo " << r.amo << ", io " << r.io << ", gen " << r.gen_in << ")" << std::endl;
  }
  CHECK(lost == 0, lost << " requests lost");
  CHECK(issued == num_reqs, "only " << issued << " requests issued");

  std::cout << "seed " << seed << ": " << issued << " requests, " << outs << " out, " << kills << " torn down, "
            << silent << " silent stores, " << walks_n << " walks, " << flushes << " flushes, " << faults
            << " fault reports, " << cycle << " cycles, " << errors << " errors" << std::endl;
  if (errors) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
