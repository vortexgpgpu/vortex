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

// Functional test for VX_fdiv_unit (F32): drives random + directed vectors through
// the fixed-latency pipeline and checks bit-exact result/fflags against the
// softfloat (rvfloats) RISC-V golden FDIV across all rounding modes. The check
// is strict (full RISC-V spec); mismatches are also bucketed by class to make
// any non-compliance actionable.

#include "VVX_fdiv_unit.h"
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

using Device = VVX_fdiv_unit;

namespace {

struct TV { uint32_t a, b, frm; };
struct Golden { uint32_t res, fflags, a, b, frm; };

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000,             // +0, -0
    0x3f800000, 0xbf800000,             // +1, -1
    0x40000000, 0xc0000000,             // +2, -2
    0x7f800000, 0xff800000,             // +inf, -inf
    0x7fc00000, 0x7fa00000, 0xffa00000, // qnan, snan, -snan
    0x00000001, 0x80000001,             // smallest subnormal +/-
    0x007fffff, 0x807fffff,             // largest subnormal +/-
    0x00800000, 0x80800000,             // smallest normal +/-
    0x7f7fffff, 0xff7fffff,             // largest normal +/-
    0x40490fdb, 0x3dcccccd, 0x41200000, 0x42c80000,
};

inline int fclass(uint32_t x) { // 0=zero 1=sub 2=norm 3=inf 4=nan
    uint32_t e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 0)    return m ? 1 : 0;
    if (e == 0xFF) return m ? 4 : 3;
    return 2;
}
// Flush subnormal -> signed zero (DAZ inputs / FTZ result for the lean config).
inline uint32_t flush_sub(uint32_t x) {
    return (((x >> 23) & 0xFF) == 0 && (x & 0x7FFFFF)) ? (x & 0x80000000u) : x;
}

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    size_t num_random = 200000;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_random = static_cast<size_t>(v);
    }

    std::vector<TV> tests;
    for (uint32_t a : kSpecials)
        for (uint32_t b : kSpecials)
            for (uint32_t f = 0; f < 5; ++f)
                tests.push_back({a, b, f});
    std::mt19937 rng(0xC0FFEEu);
    for (size_t i = 0; i < num_random; ++i)
        tests.push_back({static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()),
                         static_cast<uint32_t>(i % 5)});

#ifdef LEAN
    // SNORM/EXCEPT disabled: contract is finite, non-exceptional operands with a
    // finite result. Keep only vectors whose (DAZ'd) golden is finite and raises
    // no NV/DZ; the lean unit gives FTZ result + zero fflags for those.
    // Both SNORM and EXCEPT disabled: the exception cone (zero/sub/inf/nan) is off,
    // so the contract is NORMAL operands (the RTU's domain). Overflow/underflow
    // RESULTS are fine (of -> max/inf, FTZ -> 0). Skip the rare NV/DZ.
    { std::vector<TV> f; for (auto& t : tests) {
        if (fclass(t.a) != 2 || fclass(t.b) != 2) continue;
        uint32_t ff = 0; (void)rv_fdiv_s(t.a, t.b, t.frm, &ff);
        if (ff & 0x18) continue;
        f.push_back(t);
    } tests.swap(f); }
#endif

    const size_t N = tests.size();

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1; sim->mask = 0; sim->fmt = 0; sim->frm = 0;
    sim->dataa = 0; sim->datab = 0;
    ticks = sim.reset(ticks);

    std::deque<Golden> fifo;
    size_t feed = 0, checked = 0, errors = 0;
    size_t e_ovf = 0, e_uf = 0, e_special = 0, e_normal = 0; // mismatch buckets

    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            const TV& t = tests[feed];
            sim->dataa = t.a; sim->datab = t.b; sim->frm = t.frm; sim->fmt = 0;
            sim->mask = 1;
            uint32_t ff = 0;
            uint32_t ga = t.a, gb = t.b;
#ifdef LEAN
            ga = flush_sub(t.a); gb = flush_sub(t.b);  // DAZ inputs
#endif
            uint32_t r  = rv_fdiv_s(ga, gb, t.frm, &ff);
#ifdef LEAN
            r = flush_sub(r); ff = 0;                  // FTZ result, no fflags
#endif
            fifo.push_back({r, ff, t.a, t.b, t.frm});
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
                if (g.fflags & 0x4)       ++e_ovf;
                else if (g.fflags & 0x2)  ++e_uf;
                else if (fclass(g.a) != 2 || fclass(g.b) != 2) ++e_special;
                else ++e_normal;
                if (errors < 20)
                    printf("MISMATCH fdiv a=%08x b=%08x frm=%u : got res=%08x ff=%02x  exp res=%08x ff=%02x\n",
                           g.a, g.b, g.frm, got_r, got_f, g.res, g.fflags);
                ++errors;
            }
            ++checked;
        }
    }

    printf("fdiv_unit: checked=%zu errors=%zu  [overflow=%zu underflow=%zu special-operand=%zu normal=%zu]\n",
           checked, errors, e_ovf, e_uf, e_special, e_normal);
    printf("%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
