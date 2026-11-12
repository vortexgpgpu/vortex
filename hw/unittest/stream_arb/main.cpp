// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// Verilator unit test for VX_stream_arb over the shapes in VX_stream_arb_top.
// Each shape's harness checks its own traffic: random input valids and output
// stalls for a while, then the inputs stop and every packet must have left
// exactly once, on the right output, with the right select.

#include "vl_simulator.h"
#include "VVX_stream_arb_top.h"
#include <cstdio>
#include <cstdlib>

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

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  vl_simulator<VVX_stream_arb_top> sim;
  uint64_t tick = 0;

  constexpr uint32_t kTrafficCycles = 20000;
  constexpr uint32_t kDrainCycles   = 20000;

  sim->drain = 0;
  sim->reset = 1;
  tick = sim.step(tick, 8);
  sim->reset = 0;

  tick = sim.step(tick, 2 * kTrafficCycles);
  sim->drain = 1;
  uint32_t cycles = 0;
  while (!sim->idle && sim->errors == 0 && cycles < kDrainCycles) {
    tick = sim.step(tick, 2);
    ++cycles;
  }
  timestamp = tick;

  std::printf("delivered=%u, errors=%u, idle=%d\n", sim->delivered, sim->errors, sim->idle);
  for (uint32_t c = 0; c < 8 * sizeof(sim->busy); ++c) {
    if ((sim->busy >> c) & 1) {
      std::printf("shape %u did not drain\n", c);
    }
  }
  if (sim->errors != 0 || !sim->idle) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
