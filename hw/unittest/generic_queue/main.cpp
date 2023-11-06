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
#include "VVX_fifo_queue.h"
#include <iostream>

#define MAX_TICKS 20

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

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

using Device = VVX_fifo_queue;

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<Device> sim;

  // run test
  timestamp = sim.reset(0);
  while (timestamp < MAX_TICKS) {
    switch (timestamp) {
    case 0:
      // initial values
      sim->pop  = 0;
      sim->push = 0;
      timestamp = sim.step(timestamp, 2);
      break;
    case 2:
      // Verify outputs    
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x1);
      // push 0xa
      sim->pop     = 0;
      sim->push    = 1;
      sim->data_in = 0xa;
      break;
    case 4:
      // verify outputs    
      CHECK(sim->data_out == 0xa);
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x0);
      // push 0xb
      sim->pop     = 0;
      sim->push    = 1;
      sim->data_in = 0xb;
      break;
    case 6:
      // verify outputs    
      CHECK(sim->data_out == 0xa);
      CHECK(sim->full == 0x1);
      CHECK(sim->empty == 0x0);
      // pop
      sim->pop  = 1;
      sim->push = 0;
      break;
    case 8:
      // verify outputs    
      CHECK(sim->data_out == 0xb);
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x0);
      // pop
      sim->pop  = 1;
      sim->push = 0;
      break;
    case 10:
      // verify outputs    
      CHECK(sim->full == 0x0);
      CHECK(sim->empty == 0x1);
      sim->pop  = 0;
      sim->push = 0;
      break;
    }

    // advance clock
    timestamp = sim.step(timestamp, 2);
  }

  std::cout << "PASSED!" << std::endl;
  std::cout << "Simulation time: " << std::dec << timestamp/2 << " cycles" << std::endl;

  return 0;
}