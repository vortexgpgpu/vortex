// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// Verilator unit test for VX_alu_muldiv's synthesizable multiplier and batched
// divider: VX_alu_muldiv_top checks its own traffic, one unit per divider
// ratio; this driver clocks it until every unit has returned all of its ops.

#include "vl_simulator.h"
#include "VVX_alu_muldiv_top.h"
#include <cstdio>

static uint64_t timestamp = 0;
static bool     trace_en  = false;

double sc_time_stamp() { return timestamp; }
bool   sim_trace_enabled() { return trace_en; }
void   sim_trace_enable(bool e) { trace_en = e; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  vl_simulator<VVX_alu_muldiv_top> sim;
  uint64_t tick = 0;

  constexpr uint32_t kMaxCycles = 2000000;

  sim->reset = 1;
  tick = sim.step(tick, 8);
  sim->reset = 0;

  uint32_t cycles = 0;
  while (!sim->done && sim->errors == 0 && cycles < kMaxCycles) {
    tick = sim.step(tick, 2);
    ++cycles;
  }
  timestamp = tick;

  std::printf("cycles=%u, errors=%u, done=%d\n", cycles, sim->errors, sim->done);
  if (sim->errors != 0 || !sim->done) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
