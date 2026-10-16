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

// Bit-exact check of the VX_mufu_pe pipeline against the shared golden model
// (sw/common/mufu.cpp) for every function: special values, a strided sweep of
// the whole FP32 space, and random inputs.
//
// usage: mufu_unit [stride] [num_random]

#include "VVX_mufu_pe.h"
#include "vl_simulator.h"
#include "mufu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

using Device = VVX_mufu_pe;
using namespace vortex::mufu;

namespace {

constexpr int kLatency = 7;

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x80800000,
    0x3f800000, 0xbf800000, 0x3f000000, 0x3fc00000, 0x40000000, 0x40800000,
    0x40490fdb, 0x3eaaaaab, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
    0x7fc00000, 0xffc00000, 0x7fa00000, 0x42fe0000, 0x43000000, 0xc2fc0000,
    0xc2fe0000, 0xc3000000, 0x4b000001, 0x4b800001, 0x3f7fffff, 0x3f800001,
    0x39800000, 0x39000000, 0x41000000, 0x40ffffff, 0x3a7fffff, 0x3a800000,
};

struct Req { uint32_t fn, x; };

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    uint64_t stride = 65521;
    size_t num_random = 200000;
    if (argc > 1) stride = std::strtoull(argv[1], nullptr, 0);
    if (argc > 2) num_random = std::strtoull(argv[2], nullptr, 0);
    if (stride == 0) stride = 1;

    std::vector<Req> reqs;
    for (uint32_t fn = 0; fn < FN_COUNT; ++fn) {
        for (uint32_t x : kSpecials)
            reqs.push_back({fn, x});
        for (uint64_t x = 0; x < (1ull << 32); x += stride)
            reqs.push_back({fn, uint32_t(x)});
    }
    std::mt19937 rng(0x6d756675u);
    for (size_t i = 0; i < num_random; ++i)
        reqs.push_back({uint32_t(i % FN_COUNT), uint32_t(rng())});

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1;
    sim->fn = 0;
    sim->dataa = 0;
    ticks = sim.reset(ticks);

    std::deque<Req> inflight;
    size_t feed = 0, checked = 0;
    size_t errors[FN_COUNT] = {};
    size_t total_errors = 0;

    for (size_t c = 0; c < reqs.size() + kLatency; ++c) {
        if (feed < reqs.size()) {
            sim->fn = reqs[feed].fn;
            sim->dataa = reqs[feed].x;
            inflight.push_back(reqs[feed]);
            ++feed;
        }
        ticks = sim.step(ticks, 2);

        if (c >= size_t(kLatency - 1) && !inflight.empty()) {
            Req r = inflight.front();
            inflight.pop_front();
            uint32_t got = sim->result;
            uint32_t exp = eval(r.fn, r.x);
            if (got != exp) {
                if (total_errors < 20)
                    printf("MISMATCH %s x=%08x : got %08x expected %08x\n", fn_name(r.fn), r.x, got, exp);
                ++errors[r.fn];
                ++total_errors;
            }
            ++checked;
        }
    }

    printf("mufu_unit: checked=%zu errors=%zu [", checked, total_errors);
    for (uint32_t fn = 0; fn < FN_COUNT; ++fn)
        printf("%s%s=%zu", fn ? " " : "", fn_name(fn), errors[fn]);
    printf("]\n%s\n", total_errors ? "FAILED" : "PASSED");
    return total_errors ? 1 : 0;
}
