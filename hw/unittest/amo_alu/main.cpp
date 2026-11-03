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

// Functional test for VX_amo_alu (combinational): every op at .W and .D,
// signed and unsigned MIN/MAX, random and edge operands, against the RISC-V
// A/Zacas definitions. A .W result is the low word zero-extended.

#include "VVX_amo_alu.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#ifndef DATA_WIDTH
#define DATA_WIDTH 64
#endif
#ifndef ARITH_WIDTH
#define ARITH_WIDTH DATA_WIDTH
#endif

enum { LR = 0, SC, ADD, SWAP, XOR, OR, AND, MIN, MAX, CAS, NUM_OPS };

static uint64_t model(int op, bool uns, bool w, uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t m = w ? 0xffffffffull : ~0ull;
    a &= m; b &= m; c &= m;
    int64_t as = w ? int64_t(int32_t(a)) : int64_t(a);
    int64_t bs = w ? int64_t(int32_t(b)) : int64_t(b);
    bool lt = uns ? (a < b) : (as < bs);
    uint64_t r = 0;
    switch (op) {
    case LR:   r = a; break;
    case SC:   r = b; break;
    case SWAP: r = b; break;
    case ADD:  r = a + b; break;
    case AND:  r = a & b; break;
    case OR:   r = a | b; break;
    case XOR:  r = a ^ b; break;
    case MIN:  r = lt ? a : b; break;
    case MAX:  r = lt ? b : a; break;
    case CAS:  r = (a == c) ? b : a; break;
    }
    return r & m;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t n = 4000000;
    if (argc > 1) n = std::strtoull(argv[1], nullptr, 0);
    VVX_amo_alu dut;
    std::mt19937_64 rng(0xA30A1u);
    const uint64_t edges[] = {0, 1, 0x7fffffffull, 0x80000000ull, 0xffffffffull, 0x100000000ull,
                              0x7fffffffffffffffull, 0x8000000000000000ull, ~0ull, 0xffffffff7fffffffull,
                              0x00000001ffffffffull, 0x80000000ffffffffull};
    const int ne = sizeof(edges) / sizeof(edges[0]);
    const bool can_d = (DATA_WIDTH > 32);
    size_t errors = 0, checked = 0;
    for (size_t i = 0; i < n; ++i) {
        int op = int(rng() % NUM_OPS);
        bool w = !can_d || (rng() & 1);
        if (op != CAS && ARITH_WIDTH <= 32) w = true;
        bool uns = rng() & 1;
        uint64_t a = (i % 4 == 0) ? edges[rng() % ne] : rng();
        uint64_t b = (i % 4 == 1) ? edges[rng() % ne] : ((i % 8 == 2) ? a : rng());
        if (i % 16 == 3) b = (a & 0xffffffff00000000ull) | (rng() & 0xffffffffull); // equal high words
        uint64_t c = (i % 3 == 0) ? a : rng();
        if (DATA_WIDTH <= 32) { a &= 0xffffffffull; b &= 0xffffffffull; c &= 0xffffffffull; }
        dut.op = op; dut.is_unsigned = uns; dut.width = w ? 2 : 3;
        dut.old_word = a; dut.rhs = b; dut.cmp = c;
        dut.eval();
        uint64_t exp = model(op, uns, w, a, b, c);
        uint64_t got = dut.new_word;
        ++checked;
        if (got != exp) {
            if (errors < 16)
                printf("MISMATCH op=%d w=%d u=%d a=%016llx b=%016llx c=%016llx got=%016llx exp=%016llx\n", op, w, uns,
                       (unsigned long long)a, (unsigned long long)b, (unsigned long long)c,
                       (unsigned long long)got, (unsigned long long)exp);
            ++errors;
        }
    }
    printf("amo_alu DATA_WIDTH=%d ARITH_WIDTH=%d: checked=%zu errors=%zu\n%s\n", DATA_WIDTH, ARITH_WIDTH,
           checked, errors, errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
