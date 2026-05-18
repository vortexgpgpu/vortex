// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_launch.
//
// FSM under test:
//   IDLE         grant → PULSE_START
//   PULSE_START  one-cycle `start` pulse → WAIT_BUSY
//   WAIT_BUSY    gpu_busy ↑ → WAIT_DRAIN
//   WAIT_DRAIN   gpu_busy ↓ → done pulse → IDLE
//
// Coverage:
//   1. Reset → IDLE, no spurious start/done.
//   2. Long idle while grant=0 → no transition.
//   3. Full happy-path launch: grant → start pulse → busy rise → busy fall
//      → done pulse → back to IDLE.
//   4. Re-arm: a second launch back-to-back after done.
//   5. WAIT_BUSY hangs indefinitely until busy actually rises (no premature
//      done).
//   6. start is exactly 1 cycle wide.
//   7. done is exactly 1 cycle wide and only fires on the busy falling edge.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_launch_top.h"
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

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1); \
    } \
} while (0)

// Drive inputs, sample outputs for the current cycle, then advance one
// clock edge. Same convention used by cp_arbiter / cp_engine tests.
template <typename T>
static void cycle(vl_simulator<T>& sim, uint64_t& tick) {
    sim->eval();
    tick = sim.step(tick, 2);
}

// Run one full launch sequence and verify start/done timing. busy_hold is
// how many cycles to keep gpu_busy=1 in WAIT_DRAIN before dropping it.
template <typename T>
static void launch(vl_simulator<T>& sim, uint64_t& tick, int busy_hold) {
    // T0 IDLE with grant=1 → captures, transitions to PULSE_START at edge.
    sim->grant    = 1;
    sim->gpu_busy = 0;
    sim->eval();
    EXPECT(sim->start == 0, "start should be 0 in IDLE");
    EXPECT(sim->done  == 0, "done should be 0 in IDLE");
    cycle(sim, tick);

    // T1 PULSE_START: start asserted for exactly this cycle.
    sim->eval();
    EXPECT(sim->start == 1, "start pulse missing in PULSE_START");
    EXPECT(sim->done  == 0, "done should be 0 in PULSE_START");
    cycle(sim, tick);

    // T2 WAIT_BUSY: start back low, still no done. gpu_busy stays low for
    // a few cycles to verify we wait properly.
    sim->grant = 0;   // grant can drop now; FSM state holds
    sim->eval();
    EXPECT(sim->start == 0, "start should fall after PULSE_START");
    EXPECT(sim->done  == 0, "done in WAIT_BUSY should be 0");
    cycle(sim, tick);

    sim->eval();
    EXPECT(sim->start == 0, "start should stay 0 while waiting for busy");
    EXPECT(sim->done  == 0, "done while busy hasn't risen should be 0");
    cycle(sim, tick);

    // Drive busy=1; FSM moves to WAIT_DRAIN at next edge.
    sim->gpu_busy = 1;
    cycle(sim, tick);

    // WAIT_DRAIN with busy still high — no done yet.
    for (int i = 0; i < busy_hold; ++i) {
        sim->eval();
        EXPECT(sim->done == 0, "done fired prematurely while busy still high");
        cycle(sim, tick);
    }

    // Drop busy; this cycle WAIT_DRAIN's combinational done = (state==DRAIN) && !busy
    // fires, and at the edge FSM returns to IDLE.
    sim->gpu_busy = 0;
    sim->eval();
    EXPECT(sim->done == 1, "done should pulse on busy falling edge");
    cycle(sim, tick);

    // Back in IDLE; done falls.
    sim->eval();
    EXPECT(sim->done == 0, "done should not stick after one cycle");
    EXPECT(sim->start == 0, "start should be 0 in post-launch IDLE");
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    vl_simulator<VVX_cp_launch_top> sim;
    uint64_t tick = 0;

    sim->grant    = 0;
    sim->gpu_busy = 0;
    tick = sim.reset(tick);

    // ----- Reset & idle -----
    for (int i = 0; i < 5; ++i) {
        sim->eval();
        EXPECT(sim->start == 0, "start should be 0 during long idle");
        EXPECT(sim->done  == 0, "done should be 0 during long idle");
        cycle(sim, tick);
    }

    // ----- First launch (busy held for 1 cycle) -----
    launch(sim, tick, /*busy_hold=*/1);

    // ----- Back-to-back launch — FSM must re-arm cleanly -----
    launch(sim, tick, /*busy_hold=*/3);

    // ----- A third launch with grant pulsed only at IDLE — once captured,
    //       FSM should not require grant held high -----
    launch(sim, tick, /*busy_hold=*/0);

    std::printf("PASSED\n");
    return 0;
}
