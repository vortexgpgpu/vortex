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

// Functional test for the merged VX_fdivsqrt_unit: exercises both FDIV and
// FSQRT through the is_sqrt port and checks bit-exact result/fflags against the
// softfloat (rvfloats) RISC-V golden across all rounding modes. Strict check.
// F32 operands always; F64 operands too when built with FLEN=64.

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

#ifndef FLEN
#define FLEN 32
#endif

using Device = VVX_fdivsqrt_unit;

namespace {

struct TV { uint64_t a, b; uint32_t frm; bool sqrt, dbl; };
struct Golden { uint64_t res, a, b; uint32_t fflags, frm; bool sqrt, dbl; };

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
    0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x7f7fffff, 0xff7fffff, 0x40490fdb, 0x3dcccccd, 0x41200000, 0x42c80000,
};

// Exact squares (sqrt sets no NX), both radicand exponent parities and both
// significand ranges of the scaled radicand.
const uint32_t kSquares[] = {
    0x40800000, 0x40100000, 0x3e800000, 0x40c80000, 0x3fc80000, 0x42f20000,
    0x41100000, 0x43800000, 0x4f800000, 0x00000002,
};

#if FLEN >= 64
const uint64_t kSpecialsD[] = {
    0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000,
    0x4000000000000000, 0xc000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
    0x7ff8000000000000, 0x7ff4000000000000, 0x0000000000000001, 0x8000000000000001,
    0x000fffffffffffff, 0x800fffffffffffff, 0x0010000000000000, 0x8010000000000000,
    0x7fefffffffffffff, 0xffefffffffffffff, 0x400921fb54442d18, 0x3fb999999999999a,
    0x4024000000000000, 0x4059000000000000,
};

const uint64_t kSquaresD[] = {
    0x4010000000000000, 0x4002000000000000, 0x3fd0000000000000, 0x4019000000000000,
    0x3ff9000000000000, 0x405e400000000000, 0x4022000000000000, 0x4070000000000000,
    0x43f0000000000000, 0x0000000000000004,
};
#endif

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
    for (uint32_t a : kSpecials) {
        for (uint32_t b : kSpecials) {
            for (uint32_t f = 0; f < 5; ++f) {
                tests.push_back({a, b, f, false, false});
            }
        }
    }
    // directed SQRT
    for (uint32_t a : kSpecials) {
        for (uint32_t f = 0; f < 5; ++f) {
            tests.push_back({a, 0, f, true, false});
        }
    }
    for (uint32_t a : kSquares) {
        for (uint32_t f = 0; f < 5; ++f) {
            tests.push_back({a, 0, f, true, false});
        }
    }
    // random: alternate div / sqrt
    std::mt19937 rng(0xD1D50Fu);
    for (size_t i = 0; i < num_random; ++i) {
        bool sq = (i & 1);
        tests.push_back({static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()),
                         static_cast<uint32_t>(i % 5), sq, false});
    }
#if FLEN >= 64
    for (uint64_t a : kSpecialsD) {
        for (uint64_t b : kSpecialsD) {
            for (uint32_t f = 0; f < 5; ++f) {
                tests.push_back({a, b, f, false, true});
            }
        }
    }
    for (uint64_t a : kSpecialsD) {
        for (uint32_t f = 0; f < 5; ++f) {
            tests.push_back({a, 0, f, true, true});
        }
    }
    for (uint64_t a : kSquaresD) {
        for (uint32_t f = 0; f < 5; ++f) {
            tests.push_back({a, 0, f, true, true});
        }
    }
    std::mt19937_64 rng64(0xD1D50F64u);
    for (size_t i = 0; i < num_random; ++i) {
        bool sq = (i & 1);
        tests.push_back({rng64(), rng64(), static_cast<uint32_t>(i % 5), sq, true});
    }
#endif

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
            sim->dataa = t.a; sim->datab = t.b; sim->frm = t.frm; sim->fmt = t.dbl ? 1 : 0;
            sim->is_sqrt = t.sqrt ? 1 : 0; sim->mask = 1;
            uint32_t ff = 0;
            uint64_t r;
            if (t.dbl) {
                r = t.sqrt ? rv_fsqrt_d(t.a, t.frm, &ff)
                           : rv_fdiv_d(t.a, t.b, t.frm, &ff);
            } else {
                uint32_t a = static_cast<uint32_t>(t.a), b = static_cast<uint32_t>(t.b);
                r = t.sqrt ? rv_fsqrt_s(a, t.frm, &ff)
                           : rv_fdiv_s(a, b, t.frm, &ff);
            }
            fifo.push_back({r, t.a, t.b, ff, t.frm, t.sqrt, t.dbl});
            ++feed;
        } else {
            sim->mask = 0;
        }
        sim->enable = 1;
        ticks = sim.step(ticks, 2);

        if (c >= (size_t)(LATENCY - 1) && !fifo.empty()) {
            Golden g = fifo.front(); fifo.pop_front();
            uint64_t got_r = sim->result;
            uint32_t got_f = sim->fflags;
            if (!g.dbl) {
                got_r &= 0xffffffffull;
            }
            if (got_r != g.res || got_f != g.fflags) {
                if (g.sqrt) ++e_sqrt; else ++e_div;
                if (errors < 20) {
                    printf("MISMATCH %s.%c a=%016llx b=%016llx frm=%u : got res=%016llx ff=%02x  exp res=%016llx ff=%02x\n",
                           g.sqrt ? "fsqrt" : "fdiv ", g.dbl ? 'd' : 's',
                           (unsigned long long)g.a, (unsigned long long)g.b, g.frm,
                           (unsigned long long)got_r, got_f, (unsigned long long)g.res, g.fflags);
                }
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
