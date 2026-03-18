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

#include "vl_simulator.h"
#include "VVX_cta_dispatch_top.h"
#include <iostream>
#include "VX_config.h"
#include "VX_types.h"
#include <cassert>

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

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<VVX_cta_dispatch_top> sim;

  int tick = 0;

  tick = sim.reset(tick);

  sim->task_in_req_valid = 1;
  sim->num_warps = 4;
  sim->start_pc = 0x12345678;
  sim->input_param = 0x87654321;
  sim->input_cta_x = 1;
  sim->input_cta_y = 2;
  sim->input_cta_z = 3;
  sim->input_cta_id = 13;
  sim->input_remain_mask = 0b1100;
  sim->active_warps = 0;

  // Track per-warp countdown timers; expire after 4 cycles → warp slot freed
  std::vector<int> warp_timer(32, 0);

  for (int i = 0; i < 30; ++i) {
    tick = sim.step(tick, 2);

    // A new warp was dispatched this cycle; start its timer
    if (sim->cta_sched_fire) {
      int w = sim->cta_sched_wid;
      if (warp_timer[w] == 0) warp_timer[w] = 4;
    }

    // Advance timers; rebuild active_warps mask
    uint32_t next_active_warps = 0;
    for (int w = 0; w < 32; ++w) {
      if (warp_timer[w] > 0) {
        warp_timer[w]--;
        if (warp_timer[w] > 0)
          next_active_warps |= (1u << w);
      }
    }
    sim->active_warps = next_active_warps;
    sim->task_in_req_valid = 0;
  }

  return 0;
}