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

// Functional test for the merged VX_fdivsqrt_unit (F32): exercises both FDIV and
// FSQRT through the is_sqrt port and checks bit-exact result/fflags against the
// softfloat (rvfloats) RISC-V golden across all rounding modes. Strict check.

#include "VVX_fdivsqrt_unit.h"
#include "vl_simulator.h"
#include "rvfloats.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#ifndef LATENCY
#define LATENCY 17
#endif

using Device = VVX_fdivsqrt_unit;

namespace {

struct TV { uint32_t a, b, frm; bool sqrt; };
struct Golden { uint32_t res, fflags, a, b, frm; bool sqrt; };

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
    0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x7f7fffff, 0xff7fffff, 0x40490fdb, 0x3dcccccd, 0x41200000, 0x42c80000,
};

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    size_t num_random = 150000;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_random = static_cast<size_t>(v);
    }

    std::vector<TV> tests;
    // directed DIV pairs
    for (uint32_t a : kSpecials)
        for (uint32_t b : kSpecials)
            for (uint32_t f = 0; f < 5; ++f)
                tests.push_back({a, b, f, false});
    // directed SQRT
    for (uint32_t a : kSpecials)
        for (uint32_t f = 0; f < 5; ++f)
            tests.push_back({a, 0, f, true});
    // random: alternate div / sqrt
    std::mt19937 rng(0xD1D50Fu);
    for (size_t i = 0; i < num_random; ++i) {
        bool sq = (i & 1);
        tests.push_back({static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()),
                         static_cast<uint32_t>(i % 5), sq});
    }

    const size_t N = tests.size();

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1; sim->mask = 0; sim->fmt = 0; sim->frm = 0;
    sim->dataa = 0; sim->datab = 0; sim->is_sqrt = 0;
    ticks = sim.reset(ticks);

    std::deque<Golden> fifo;
    size_t feed = 0, checked = 0, errors = 0, e_div = 0, e_sqrt = 0;

    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            const TV& t = tests[feed];
            sim->dataa = t.a; sim->datab = t.b; sim->frm = t.frm; sim->fmt = 0;
            sim->is_sqrt = t.sqrt ? 1 : 0; sim->mask = 1;
            uint32_t ff = 0;
            uint32_t r  = t.sqrt ? rv_fsqrt_s(t.a, t.frm, &ff)
                                 : rv_fdiv_s(t.a, t.b, t.frm, &ff);
            fifo.push_back({r, ff, t.a, t.b, t.frm, t.sqrt});
            ++feed;
        } else {
            sim->mask = 0;
        }
        sim->enable = 1;
        ticks = sim.step(ticks, 2);

        if (c >= (size_t)(LATENCY - 1) && !fifo.empty()) {
            Golden g = fifo.front(); fifo.pop_front();
            uint32_t got_r = sim->result, got_f = sim->fflags;
            if (got_r != g.res || got_f != g.fflags) {
                if (g.sqrt) ++e_sqrt; else ++e_div;
                if (errors < 20)
                    printf("MISMATCH %s a=%08x b=%08x frm=%u : got res=%08x ff=%02x  exp res=%08x ff=%02x\n",
                           g.sqrt ? "fsqrt" : "fdiv ", g.a, g.b, g.frm, got_r, got_f, g.res, g.fflags);
                ++errors;
            }
            ++checked;
        }
    }

    printf("fdivsqrt: checked=%zu errors=%zu  [div=%zu sqrt=%zu]\n",
           checked, errors, e_div, e_sqrt);
    printf("%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
