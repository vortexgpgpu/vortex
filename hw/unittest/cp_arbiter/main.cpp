// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_arbiter (round-robin over 4 bidders).
//
// Coverage:
//   1. Single bidder asserts: gets every cycle.
//   2. All bidders assert continuously: each wins every 4th cycle in turn.
//   3. Bidder activity changes mid-stream: rotation skips inactive bidders
//      but advances past the last winner so the schedule stays fair.
//   4. Reset behavior: rr_ptr returns to 0; first cycle after release picks
//      the lowest-indexed valid bidder.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_arbiter_top.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

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

// 4-bit packed grant -> which bidder index won (or -1 for none, -2 for >1).
static int winner_of(uint8_t g) {
    int w = -1;
    for (int i = 0; i < 4; ++i) if (g & (1u << i)) {
        if (w >= 0) return -2;
        w = i;
    }
    return w;
}

#define EXPECT(cond, msg) do {                                          \
    if (!(cond)) {                                                      \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1);                                                   \
    }                                                                   \
} while (0)

// Drive new inputs, sample the *current cycle's* grant (combinational on
// the pre-edge rr_ptr state), THEN advance the clock so the FF latches
// for the next cycle. Reading after step(2) would observe the
// combinational re-evaluation with the *new* rr_ptr, i.e. one cycle in
// the future — which makes the rotation off-by-one and hard to reason
// about. Sampling first matches the natural "this cycle's winner" view.
template <typename T>
static uint8_t tick_with_inputs(vl_simulator<T>& sim, uint64_t& tick,
                                uint8_t valid, uint8_t prio_pack) {
    sim->bid_valid    = valid;
    sim->bid_priority = prio_pack;
    sim->eval();
    uint8_t g = sim->bid_grant;
    tick = sim.step(tick, 2);   // commit the clock edge for next call
    return g;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    vl_simulator<VVX_cp_arbiter_top> sim;
    uint64_t tick = 0;
    tick = sim.reset(tick);

    // ----- Test 1: single bidder, bid 2 only -----
    for (int cyc = 0; cyc < 5; ++cyc) {
        uint8_t g = tick_with_inputs(sim, tick, /*valid=*/0b0100, 0);
        EXPECT(winner_of(g) == 2, "single bidder should always win");
    }

    // Idle one cycle so rr_ptr lands at a known position. After test 1,
    // rr_ptr is at 3 (one past the last winner 2). The idle cycle has no
    // grant, so rr_ptr stays.
    tick_with_inputs(sim, tick, 0, 0);

    // ----- Test 2: all four bidders, observe round-robin over 8 cycles. -----
    // rr_ptr at this point = 3 (from test 1). So first winner should be 3,
    // then 0, 1, 2, 3, 0, ...
    int expected_seq[8] = {3, 0, 1, 2, 3, 0, 1, 2};
    for (int cyc = 0; cyc < 8; ++cyc) {
        uint8_t g = tick_with_inputs(sim, tick, /*valid=*/0b1111, 0);
        int w = winner_of(g);
        if (w != expected_seq[cyc]) {
            std::fprintf(stderr,
                "FAIL T2 cycle %d: expected winner %d, got %d (grant=0x%x)\n",
                cyc, expected_seq[cyc], w, g);
            return 1;
        }
    }

    // ----- Test 3: valid bidders change mid-stream. -----
    // Keep only bidders {1,3} live. rr_ptr is at 3 now (one past winner 2).
    // First cycle: 3 valid -> grant 3. rr_ptr -> 0. Next cycle: skip 0
    // (invalid), grant 1. rr_ptr -> 2. Next: skip 2, grant 3. ...
    int expected_alt[6] = {3, 1, 3, 1, 3, 1};
    for (int cyc = 0; cyc < 6; ++cyc) {
        uint8_t g = tick_with_inputs(sim, tick, /*valid=*/0b1010, 0);
        int w = winner_of(g);
        if (w != expected_alt[cyc]) {
            std::fprintf(stderr,
                "FAIL alt cycle %d: expected %d got %d (grant=0x%x)\n",
                cyc, expected_alt[cyc], w, g);
            return 1;
        }
    }

    // ----- Test 4: no bidder valid -> no grant. -----
    for (int cyc = 0; cyc < 3; ++cyc) {
        uint8_t g = tick_with_inputs(sim, tick, /*valid=*/0, 0);
        EXPECT(g == 0, "no grant when no bidders are valid");
    }

    // ----- Test 5: reset returns rr_ptr to 0. After reset, with valid=0b1111,
    // first winner must be 0 (not whatever it would have been from prior state).
    tick = sim.reset(tick);
    {
        uint8_t g = tick_with_inputs(sim, tick, /*valid=*/0b1111, 0);
        int w = winner_of(g);
        EXPECT(w == 0, "after reset, first valid bidder is 0");
    }

    std::printf("PASSED\n");
    return 0;
}
