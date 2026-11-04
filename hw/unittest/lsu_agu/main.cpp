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

// Functional test for the two-stage LSU AGU (VX_lsu_agu + VX_lsu_agu_swz)
// against the single-cycle oracle: plain and pack-load forms, random
// addresses and addresses around and inside the stack window.

#include "VVX_lsu_agu_top.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t n = 4000000;
    if (argc > 1) n = std::strtoull(argv[1], nullptr, 0);
    VVX_lsu_agu_top dut;
    dut.eval();
    const uint64_t bot = dut.win_bottom, top = dut.win_top, span = top - bot;
    std::mt19937_64 rng(0xA6Bu);
    size_t errors = 0, in_window = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t base;
        switch (i % 4) {
        case 0: base = rng(); break;
        case 1: base = bot + (rng() % span); break;                       // inside
        case 2: base = bot - 4096 + (rng() % 8192); break;                // bottom edge
        default: base = top - 4096 + (rng() % 8192); break;               // top edge
        }
        dut.base   = base;
        dut.stride = (i % 3 == 0) ? rng() : (rng() % 65536);
        dut.offset = rng() & 0xfff;
        dut.pack   = (i % 5 == 0) ? (rng() & 3) : 0;
        dut.eval();
        in_window += (dut.lin_ref >= bot && dut.lin_ref < top);
        if (dut.addr_new != dut.addr_ref || dut.lin_new != dut.lin_ref) {
            if (errors < 16)
                printf("MISMATCH base=%016llx stride=%016llx off=%03x pack=%u: addr %016llx/%016llx lin %016llx/%016llx\n",
                       (unsigned long long)base, (unsigned long long)dut.stride, dut.offset, dut.pack,
                       (unsigned long long)dut.addr_new, (unsigned long long)dut.addr_ref,
                       (unsigned long long)dut.lin_new, (unsigned long long)dut.lin_ref);
            ++errors;
        }
    }
    printf("lsu_agu window=[%016llx,%016llx): checked=%zu in_window=%zu errors=%zu\n%s\n",
           (unsigned long long)bot, (unsigned long long)top, n, in_window, errors, errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
