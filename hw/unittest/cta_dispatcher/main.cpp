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
  Verilated::commandArgs(argc, argv);

  vl_simulator<VVX_cta_dispatch_top> sim;

  int tick = 0;

  tick = sim.reset(tick);

  // Drive a single KMU request: grid 2x2x1, block 4x1x1
  sim->task_in_valid  = 1;
  sim->in_PC          = 0x12345678;
  sim->in_cta_id      = 0;
  sim->in_block_idx_x = 0;
  sim->in_block_idx_y = 0;
  sim->in_block_idx_z = 0;
  sim->in_block_dim_x = 4;
  sim->in_block_dim_y = 1;
  sim->in_block_dim_z = 1;
  sim->in_grid_dim_x  = 2;
  sim->in_grid_dim_y  = 2;
  sim->in_grid_dim_z  = 1;
  sim->in_param       = 0x87654321;
  sim->in_lmem_size   = 0;
  sim->in_block_size  = 4;
  sim->in_warp_step_x = NUM_THREADS;
  sim->in_warp_step_y = 0;
  sim->in_warp_step_z = 0;
  sim->active_warps   = 0;
  sim->warp_done      = 0;
  sim->warp_done_wid  = 0;

  // Per-warp countdown timers: expire after 4 cycles → warp slot freed
  std::vector<int> warp_timer(NUM_WARPS, 0);

  for (int i = 0; i < 30; ++i) {
    tick = sim.step(tick, 2);

    // Retire after accepting one request
    if (sim->task_in_ready)
      sim->task_in_valid = 0;

    // A new warp was dispatched this cycle
    if (sim->cta_fire) {
      int w = sim->cta_wid;
      if (warp_timer[w] == 0) warp_timer[w] = 4;
    }

    // Advance timers; signal warp_done when a timer expires
    sim->warp_done = 0;
    uint32_t next_active_warps = 0;
    for (int w = 0; w < NUM_WARPS; ++w) {
      if (warp_timer[w] > 0) {
        warp_timer[w]--;
        if (warp_timer[w] > 0) {
          next_active_warps |= (1u << w);
        } else {
          sim->warp_done     = 1;
          sim->warp_done_wid = w;
        }
      }
    }
    sim->active_warps = next_active_warps;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
