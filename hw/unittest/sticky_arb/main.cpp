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

// Sticky ("greedy") arbitration contract, for every VX_generic_arbiter policy:
// a grant is held only while its request stays asserted. Once the request
// drops, a later request competes under the base policy.
//  1. hold:    a grantee that keeps requesting keeps the grant.
//  2. release: a grantee that stopped requesting (one idle cycle, nothing
//              granted) does not take the grant back from a requester the
//              base policy prefers.
//  3. progress: requests that only appear together in short windows (the
//              scoreboard's LSU-credit pattern: all waiting warps see a freed
//              dispatch credit in the same cycle) reach every requester within
//              NUM_REQS windows. Not checked for "P" (fixed priority).
// Before the fix, prev_grant survived the idle gap: the last grantee won every
// window and the other requesters starved.

#include <cstdio>
#include <cstdint>
#include <vector>
#include "VVX_sticky_arb_top.h"
#include "vl_simulator.h"

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif
#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool     trace_en  = false;

double sc_time_stamp() { return timestamp; }
bool   sim_trace_enabled() { return trace_en; }
void   sim_trace_enable(bool e) { trace_en = e; }

static constexpr uint32_t kNumReqs  = 16;
static constexpr uint32_t kNumArbs  = 5;
static constexpr uint32_t kWindows  = 256;
static constexpr uint32_t kGap      = 7; // idle cycles between windows
static const char* kNames[kNumArbs] = {"P", "R", "M", "C", "G"};

static vl_simulator<VVX_sticky_arb_top> sim;
static uint64_t tick = 0;
static uint32_t errors = 0;

static void grants(uint32_t out[kNumArbs]) {
  out[0] = sim->grant_index_p;
  out[1] = sim->grant_index_r;
  out[2] = sim->grant_index_m;
  out[3] = sim->grant_index_c;
  out[4] = sim->grant_index_g;
}

// Apply requests, read the grants of this cycle, then clock.
static void cycle(uint32_t requests, uint32_t out[kNumArbs]) {
  sim->requests    = requests;
  sim->grant_ready = 1;
  sim->eval();
  grants(out);
  if (requests != 0) {
    for (uint32_t a = 0; a < kNumArbs; ++a) {
      if (((sim->grant_valid >> a) & 1) == 0) {
        std::printf("arbiter %s: no grant for requests 0x%04x\n", kNames[a], requests);
        ++errors;
      }
    }
  }
  tick = sim.step(tick, 2);
}

static void expect_all(const char* what, const uint32_t got[kNumArbs], uint32_t want) {
  for (uint32_t a = 0; a < kNumArbs; ++a) {
    if (got[a] != want) {
      std::printf("arbiter %s: %s: granted %u, expected %u\n", kNames[a], what, got[a], want);
      ++errors;
    }
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint32_t g[kNumArbs];

  sim->requests = 0;
  sim->grant_ready = 0;
  sim->reset = 1;
  tick = sim.step(tick, 8);
  sim->reset = 0;
  tick = sim.step(tick, 2);

  // 1. hold: requester 3 alone, then 1 and 3 with 3 never dropping.
  cycle(1u << 3, g);
  expect_all("first grant", g, 3);
  cycle((1u << 1) | (1u << 3), g);
  expect_all("hold while requesting", g, 3);

  // 2. release: 3 drops for a cycle (no grant), then 1 and 3 ask together.
  //    Every base policy prefers 1 here (P: lower index; R/C: 1 is after 3 in
  //    the rotation; M/G: 3 was granted last).
  cycle(0, g);
  cycle((1u << 1) | (1u << 3), g);
  expect_all("no hold after the request dropped", g, 1);

  // 3. progress: every requester always waits; requests are visible only in a
  //    one-cycle window every kGap+1 cycles.
  for (uint32_t i = 0; i < kGap; ++i) cycle(0, g);
  std::vector<std::vector<uint32_t>> last(kNumArbs, std::vector<uint32_t>(kNumReqs, 0));
  std::vector<uint32_t> max_wait(kNumArbs, 0);
  for (uint32_t w = 1; w <= kWindows; ++w) {
    cycle((1u << kNumReqs) - 1, g);
    for (uint32_t a = 0; a < kNumArbs; ++a) {
      last[a][g[a]] = w;
    }
    for (uint32_t a = 1; a < kNumArbs; ++a) {
      for (uint32_t r = 0; r < kNumReqs; ++r) {
        uint32_t wait = w - last[a][r];
        if (wait > max_wait[a]) max_wait[a] = wait;
      }
    }
    for (uint32_t i = 0; i < kGap; ++i) cycle(0, g);
  }
  for (uint32_t a = 1; a < kNumArbs; ++a) {
    std::printf("arbiter %s: max wait %u windows\n", kNames[a], max_wait[a]);
    if (max_wait[a] > kNumReqs) {
      std::printf("arbiter %s: a requester waited %u windows (> %u)\n", kNames[a], max_wait[a], kNumReqs);
      ++errors;
    }
  }

  timestamp = tick;
  if (errors != 0) {
    std::printf("FAILED! (%u errors)\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
