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

// CTA dispatcher test: random CTAs (1D/2D blocks, partial last warps, shared
// memory footprints, context switches) against random warp lifetimes, with
// every launch checked against a model of the dispatch rules:
//  - a launched warp is free (not active) and belongs to its lane (wid % lanes);
//  - a CTA launches ceil(block_size / NUM_THREADS) warps, ranks 0..n-1 once,
//    rank r full unless it is the last of a partial block;
//  - each lane's thread indices are the rank's, linearized over block_dim;
//  - the lanes launched in a cycle are ranked in round-robin order from a
//    pointer that advances by the number launched;
//  - a warp asks for the startup stub exactly when its mask holds a lane the
//    slot has not run in this context;
//  - every CTA completes.

#include "vl_simulator.h"
#include "VVX_cta_dispatch_top.h"
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

static constexpr int clog2(int n) { int r = 0; while ((1 << r) < n) ++r; return r; }

static constexpr int NW = VX_CFG_NUM_WARPS;
static constexpr int NT = VX_CFG_NUM_THREADS;
static constexpr int NL = VX_CFG_FETCH_WIDTH;
static constexpr int NW_WIDTH = clog2(NW) > 0 ? clog2(NW) : 1;
static constexpr int TID_W = (clog2(NW) + clog2(NT)) > 0 ? (clog2(NW) + clog2(NT)) : 1;

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

struct Cta {
  int bs, bdx, bdy, nwarps, launched = 0, recorded = 0;
  std::vector<int> rank_seen;
};

static int errors = 0;
#define CHECK(cond, msg) do { if (!(cond)) { if (errors < 20) std::cout << "cycle " << cycle << ": " << msg << std::endl; ++errors; } } while (0)

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  unsigned seed = 1;
  int num_ctas = 2000;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--seed=", 0) == 0) seed = std::stoul(a.substr(7));
    if (a.rfind("--ctas=", 0) == 0) num_ctas = std::stoi(a.substr(7));
  }
  std::mt19937 rng(seed);

  vl_simulator<VVX_cta_dispatch_top> sim;
  uint64_t tick = 0;

  sim->task_in_valid  = 0;
  sim->in_PC          = 0x12345678;
  sim->in_cta_id      = 0;
  sim->in_block_idx_x = 0;
  sim->in_block_idx_y = 0;
  sim->in_block_idx_z = 0;
  sim->in_block_dim_z = 1;
  sim->in_grid_dim_x  = 1;
  sim->in_grid_dim_y  = 1;
  sim->in_grid_dim_z  = 1;
  sim->in_param       = 0x87654321;
  sim->in_warp_step_z = 0;
  sim->in_ctx_id      = 0;
  sim->active_warps   = 0;
  sim->warp_done      = 0;
  sim->warp_done_wid  = 0;
  tick = sim.reset(tick);

  // warp state: 0 free, >0 cycles to live, -1 retired but not yet reported
  std::vector<int> life(NW, 0);
  std::deque<int> done_q;
  std::vector<int> warp_cta(NW, -1);
  std::vector<uint64_t> warp_tmask(NW, 0);
  std::vector<int> fire_group(NW, -1);
  std::vector<uint64_t> init_lanes(NW, 0);
  std::vector<Cta> ctas;
  // per launch cycle: the lanes in round-robin order from the model pointer,
  // and their warps; checked once every rank of the group is recorded
  std::map<int, std::vector<int>> groups;
  std::map<int, int> group_left;
  int rr = 0, next_group = 0;
  int cur_ctx = 0, req_ctx = 0;
  bool pending = false;
  int accepted = 0, completed_warps = 0, total_warps = 0;
  const uint64_t all_t = (NT >= 64) ? ~0ull : ((1ull << NT) - 1);

  int cycle = 0;
  for (; cycle < 4000000; ++cycle) {
    if (!pending && accepted < num_ctas && (rng() % 3 == 0)) {
      int bs;
      switch (rng() % 4) {
      case 0:  bs = 1 + rng() % NT; break;
      case 1:  bs = 1 + rng() % (NL * NT * 2); break;
      case 2:  bs = 1 + rng() % (NW * NT); break;
      default: bs = NT * (1 + rng() % NW); break;
      }
      if (bs > NW * NT) bs = NW * NT;
      int bdx = bs, bdy = 1;
      if (rng() % 3 == 0) {
        for (int d = 2; d <= 16; ++d) {
          if (bs % d == 0 && (rng() % 2)) { bdx = bs / d; bdy = d; break; }
        }
      }
      sim->in_block_dim_x = bdx;
      sim->in_block_dim_y = bdy;
      sim->in_block_size  = bs;
      sim->in_warp_step_x = NT % bdx;
      sim->in_warp_step_y = (NT / bdx) % bdy;
      sim->in_lmem_size   = (rng() % 3 == 0) ? (rng() % (1 << VX_CFG_LMEM_LOG_SIZE)) : 0;
      if (rng() % 40 == 0) req_ctx = (req_ctx + 1) & 0xff;
      sim->in_ctx_id      = req_ctx;
      sim->task_in_valid  = 1;
      pending = true;
      Cta c; c.bs = bs; c.bdx = bdx; c.bdy = bdy; c.nwarps = (bs + NT - 1) / NT;
      c.rank_seen.assign(c.nwarps, -1);
      ctas.push_back(c);
    }

    // one retirement per cycle; a warp stays active until it is reported
    sim->warp_done = 0;
    if (!done_q.empty() && (rng() % 4 != 0)) {
      int w = done_q.front(); done_q.pop_front();
      sim->warp_done = 1;
      sim->warp_done_wid = w;
      life[w] = 0;
      ++completed_warps;
    }
    for (int w = 0; w < NW; ++w) set_bits(sim->active_warps, w, 1, life[w] != 0);

    sim->eval();
    bool accept = sim->task_in_valid && sim->task_in_ready;
    tick = sim.step(tick, 2);
    ++timestamp;
    if (accept) {
      sim->task_in_valid = 0;
      pending = false;
      ++accepted;
      if (req_ctx != cur_ctx) {
        cur_ctx = req_ctx;
        std::fill(init_lanes.begin(), init_lanes.end(), 0);
      }
    }

    for (int w = 0; w < NW; ++w) {
      if (life[w] > 0 && --life[w] == 0) { life[w] = -1; done_q.push_back(w); }
    }

    // launches (registered outputs, visible after the edge)
    std::vector<int> fired;
    for (int o = 0; o < NL; ++o) {
      int l = (rr + o) % NL;
      if (!bits(sim->cta_fire, l, 1)) continue;
      int w = int(bits(sim->cta_wid, l * NW_WIDTH, NW_WIDTH));
      uint64_t tm = bits(sim->cta_tmask, l * NT, NT);
      int c = accepted - 1;
      CHECK(c >= 0, "launch before any CTA");
      CHECK(w % NL == l, "lane " << l << " launched warp " << w);
      CHECK(life[w] == 0, "warp " << w << " launched while active");
      CHECK(tm != 0 && ((tm & (tm + 1)) == 0), "tmask not a prefix mask");
      bool need_init = (tm & ~init_lanes[w]) != 0;
      CHECK(bool(bits(sim->cta_init, l, 1)) == need_init, "warp " << w << " init " << bits(sim->cta_init, l, 1) << " expected " << need_init);
      init_lanes[w] |= tm;
      // a warp outlives its record's write (TID_STAGES <= NUM_THREADS cycles)
      life[w] = NT + 2 + rng() % ((rng() % 4 == 0) ? 300 : 40);
      warp_cta[w] = c;
      warp_tmask[w] = tm;
      fire_group[w] = next_group;
      fired.push_back(w);
      if (c >= 0) ++ctas[c].launched;
      ++total_warps;
    }
    if (!fired.empty()) {
      groups[next_group] = fired;
      group_left[next_group] = int(fired.size());
      ++next_group;
      rr = (rr + int(fired.size())) % NL;
    }

    // warp records
    for (int l = 0; l < NL; ++l) {
      if (!bits(sim->rec_write, l, 1)) continue;
      int w = int(bits(sim->rec_wid, l * NW_WIDTH, NW_WIDTH));
      int r = int(bits(sim->rec_rank, l * NW_WIDTH, NW_WIDTH));
      int c = warp_cta[w];
      CHECK(c >= 0 && fire_group[w] >= 0, "record of warp " << w << " never launched");
      if (c < 0 || fire_group[w] < 0) continue;
      Cta& cta = ctas[c];
      CHECK(r < cta.nwarps, "rank " << r << " of " << cta.nwarps);
      if (r < cta.nwarps) {
        CHECK(cta.rank_seen[r] < 0, "rank " << r << " launched twice");
        cta.rank_seen[r] = w;
      }
      bool full = (r < cta.bs / NT);
      uint64_t exp_tm = full ? all_t : ((1ull << (cta.bs % NT)) - 1);
      CHECK(warp_tmask[w] == exp_tm, "rank " << r << " tmask " << std::hex << warp_tmask[w] << " expected " << exp_tm << std::dec);
      for (int i = 0; i < NT; ++i) {
        if (!((exp_tm >> i) & 1)) continue;
        int lin = r * NT + i;
        int x = lin % cta.bdx, y = (lin / cta.bdx) % cta.bdy, z = lin / (cta.bdx * cta.bdy);
        uint64_t tid = bits(sim->rec_tid, (l * NT + i) * 3 * TID_W, 3 * TID_W);
        uint64_t m = (1ull << TID_W) - 1;
        CHECK((tid & m) == uint64_t(x) && ((tid >> TID_W) & m) == uint64_t(y) && ((tid >> (2 * TID_W)) & m) == uint64_t(z),
              "rank " << r << " thread " << i << " index");
      }
      ++cta.recorded;
      int g = fire_group[w];
      if (--group_left[g] == 0) {
        // ranks follow round-robin order: groups[g] is in that order
        std::vector<int> rk;
        for (int gw : groups[g]) {
          int rw = -1;
          for (int k = 0; k < cta.nwarps; ++k) if (cta.rank_seen[k] == gw) rw = k;
          rk.push_back(rw);
        }
        for (size_t k = 1; k < rk.size(); ++k) {
          CHECK(rk[k] == rk[k - 1] + 1, "launch group out of round-robin order");
        }
        groups.erase(g); group_left.erase(g);
      }
      fire_group[w] = -1;
    }

    if (accepted == num_ctas && !pending && !sim->busy && done_q.empty()) {
      bool idle = true;
      for (int w = 0; w < NW; ++w) idle &= (life[w] == 0);
      if (idle && groups.empty()) break;
    }
  }

  CHECK(accepted == num_ctas, "only " << accepted << " of " << num_ctas << " CTAs accepted");
  for (size_t c = 0; c < ctas.size() && c < size_t(accepted); ++c) {
    CHECK(ctas[c].launched == ctas[c].nwarps, "CTA " << c << " launched " << ctas[c].launched << " of " << ctas[c].nwarps);
    CHECK(ctas[c].recorded == ctas[c].nwarps, "CTA " << c << " recorded " << ctas[c].recorded << " of " << ctas[c].nwarps);
  }

  std::cout << "lanes " << NL << ", warps " << NW << ", threads " << NT << ", seed " << seed
            << ": " << accepted << " CTAs, " << total_warps << " warps, " << cycle << " cycles, "
            << errors << " errors" << std::endl;
  if (errors) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
