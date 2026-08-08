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
#include "VVX_kmu_arb_top.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static uint64_t timestamp = 0;
static bool trace_enabled = false;

double sc_time_stamp() {
    return timestamp;
}

bool sim_trace_enabled() {
    return trace_enabled;
}

void sim_trace_enable(bool enable) {
    trace_enabled = enable;
}

#define EXPECT(cond, msg) do {                                             \
    if (!(cond)) {                                                         \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    vl_simulator<VVX_kmu_arb_top> sim;
    uint64_t tick = 0;

    sim->in_valid = 0;
    sim->out_ready = 0;
    tick = sim.reset(tick);
    sim->eval();

    EXPECT(!sim->pending, "pending must be low after reset");
    EXPECT(sim->out_valid == 0, "no output may be valid after reset");

    // An input that cannot yet enter the fanout is still an in-flight launch.
    sim->in_valid = 1;
    sim->out_ready = 0;
    sim->eval();
    EXPECT(sim->pending, "a stalled input request must assert pending");
    EXPECT(!sim->in_ready, "input must stall while every destination is blocked");

    sim->in_valid = 0;
    sim->eval();
    EXPECT(!sim->pending, "pending must clear when the stalled input is withdrawn");

    // Let the registered fanout accept one request, then backpressure the
    // selected destination. This is the handoff that used to create a false
    // device-idle cycle after the source dropped its own busy indication.
    sim->in_valid = 1;
    sim->out_ready = 0x3;
    sim->eval();
    EXPECT(sim->pending, "the presented request must assert pending");
    EXPECT(sim->in_ready, "one destination must be able to accept the request");

    tick = sim.step(tick, 2);
    sim->in_valid = 0;
    sim->out_ready = 0;
    sim->eval();

    const uint8_t held_output = sim->out_valid;
    EXPECT(held_output == 0x1 || held_output == 0x2,
           "exactly one output must hold the buffered request");
    std::printf("buffered handoff: in_valid=%u out_valid=0x%x pending=%u\n",
                static_cast<unsigned>(sim->in_valid),
                static_cast<unsigned>(held_output),
                static_cast<unsigned>(sim->pending));
    EXPECT(sim->pending,
           "pending must bridge the source-to-buffer ownership transfer");

    // The request can remain buffered for an arbitrary number of cycles.
    // pending must not pulse or depend on downstream readiness.
    for (int cycle = 0; cycle < 3; ++cycle) {
        tick = sim.step(tick, 2);
        sim->eval();
        EXPECT(sim->out_valid == held_output,
               "the same output must retain the backpressured request");
        EXPECT(sim->pending,
               "a backpressured buffered request must keep pending asserted");
    }

    // Accept the held request. pending remains high in the transfer cycle and
    // clears only after the destination has taken ownership.
    sim->out_ready = held_output;
    sim->eval();
    EXPECT((sim->out_valid & sim->out_ready) == held_output,
           "the selected destination must complete the handshake");
    EXPECT(sim->pending, "pending must cover the output handshake cycle");

    tick = sim.step(tick, 2);
    sim->out_ready = 0;
    sim->eval();
    EXPECT(sim->out_valid == 0, "the request must leave the fanout after acceptance");
    EXPECT(!sim->pending, "pending must clear after the fanout drains");

    std::printf("PASSED: KMU fanout pending covers input, buffering, and output\n");
    return 0;
}
