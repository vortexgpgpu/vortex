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

// Functional test for VX_fsqrt_unit (F32): drives random + directed vectors through
// the fixed-latency pipeline and checks bit-exact result/fflags against the
// softfloat (rvfloats) RISC-V golden FSQRT across all rounding modes. Strict
// check; mismatches bucketed by class for actionability.

#include "VVX_fsqrt_unit.h"
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

using Device = VVX_fsqrt_unit;

namespace {

struct TV { uint32_t a, frm; };
struct Golden { uint32_t res, fflags, a, frm; };

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0x40800000,
    0x40490fdb, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
    0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x7f7fffff, 0xff7fffff, 0x3dcccccd, 0x41200000, 0x42c80000,
};

inline int fclass(uint32_t x) { // 0=zero 1=sub 2=norm 3=inf 4=nan
    uint32_t e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 0)    return m ? 1 : 0;
    if (e == 0xFF) return m ? 4 : 3;
    return 2;
}
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
        for (uint32_t f = 0; f < 5; ++f)
            tests.push_back({a, f});
    std::mt19937 rng(0x5EED1234u);
    for (size_t i = 0; i < num_random; ++i)
        tests.push_back({static_cast<uint32_t>(rng()), static_cast<uint32_t>(i % 5)});

#ifdef LEAN
    // lean: finite, non-exceptional radicands with finite result; FTZ + no fflags.
    { std::vector<TV> f; for (auto& t : tests) {
        if (fclass(t.a) != 2) continue;  // normal operands only (RTU domain)
        uint32_t ff = 0; (void)rv_fsqrt_s(t.a, t.frm, &ff);
        if (ff & 0x18) continue;         // skip NV (negative radicand)
        f.push_back(t);
    } tests.swap(f); }
#endif

    const size_t N = tests.size();

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1; sim->mask = 0; sim->fmt = 0; sim->frm = 0; sim->dataa = 0;
    ticks = sim.reset(ticks);

    std::deque<Golden> fifo;
    size_t feed = 0, checked = 0, errors = 0;
    size_t e_uf = 0, e_subin = 0, e_special = 0, e_normal = 0; // mismatch buckets

    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            const TV& t = tests[feed];
            sim->dataa = t.a; sim->frm = t.frm; sim->fmt = 0; sim->mask = 1;
            uint32_t ff = 0;
            uint32_t ga = t.a;
#ifdef LEAN
            ga = flush_sub(t.a);
#endif
            uint32_t r  = rv_fsqrt_s(ga, t.frm, &ff);
#ifdef LEAN
            r = flush_sub(r); ff = 0;
#endif
            fifo.push_back({r, ff, t.a, t.frm});
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
                if (g.fflags & 0x2)            ++e_uf;
                else if (fclass(g.a) == 1)     ++e_subin;
                else if (fclass(g.a) != 2)     ++e_special;
                else ++e_normal;
                if (errors < 20)
                    printf("MISMATCH fsqrt a=%08x frm=%u : got res=%08x ff=%02x  exp res=%08x ff=%02x\n",
                           g.a, g.frm, got_r, got_f, g.res, g.fflags);
                ++errors;
            }
            ++checked;
        }
    }

    printf("fsqrt_unit: checked=%zu errors=%zu  [underflow=%zu subnormal-input=%zu special-operand=%zu normal=%zu]\n",
           checked, errors, e_uf, e_subin, e_special, e_normal);
    printf("%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
