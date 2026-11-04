// Copyright © 2019-2026
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

// Functional test for VX_mem_req_pending: a section that holds requests,
// fed and drained at random (several inputs and outputs a cycle, requests
// passing through in the cycle they enter, long idle and drain phases).
// empty must never be true while a request is held after the previous
// cycle's handshakes, and must be true when the section is empty and no
// input fired the previous cycle.

#include "VVX_mem_req_pending.h"
#include "vl_simulator.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#ifndef NI
#define NI 9
#endif
#ifndef NO
#define NO 1
#endif

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t cycles = 2000000;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) cycles = static_cast<size_t>(v);
    }
    vl_simulator<VVX_mem_req_pending> sim;
    std::mt19937 rng(0x9E4Du);
    const int cap = 8 * NO + 2 * NI;

    sim->in_fire = 0; sim->out_fire = 0;
    uint64_t ticks = sim.reset(0);

    int held = 0;             // requests held after the last cycle's handshakes
    bool in_last = false;     // an input fired last cycle
    size_t false_empty = 0, missed = 0, empties = 0, late = 0;
    for (size_t c = 0; c < cycles; ++c) {
        const bool e = sim->empty;
        if (e && held != 0) { if (false_empty < 16) printf("FALSE EMPTY cycle %zu held=%d\n", c, held); ++false_empty; }
        if (!e && held == 0 && !in_last) { if (missed < 16) printf("MISSED EMPTY cycle %zu\n", c); ++missed; }
        if (!e && held == 0 && in_last) ++late;
        empties += e;

        const int phase = (c / 1024) % 4;   // fill, steady, drain, idle
        const int p_in  = phase == 0 ? 40 : (phase == 1 ? 15 : (phase == 2 ? 2 : 0));
        const int p_out = phase == 0 ? 20 : (phase == 1 ? 60 : 80);
        uint32_t in = 0, out = 0;
        int nin = 0;
        for (int i = 0; i < NI; ++i)
            if (held + nin < cap && int(rng() % 100) < p_in) { in |= 1u << i; ++nin; }
        int avail = held + nin;      // a request may leave in the cycle it enters
        int nout = 0;
        for (int i = 0; i < NO; ++i)
            if (nout < avail && int(rng() % 100) < p_out) { out |= 1u << i; ++nout; }
        sim->in_fire = in; sim->out_fire = out;
        ticks = sim.step(ticks, 2);
        held += nin - nout;
        in_last = (nin != 0);
    }
    printf("mem_req_pending NI=%d NO=%d: cycles=%zu empty=%zu late=%zu false_empty=%zu missed=%zu\n%s\n",
           NI, NO, cycles, empties, late, false_empty, missed, (false_empty || missed) ? "FAILED" : "PASSED");
    return (false_empty || missed) ? 1 : 0;
}
