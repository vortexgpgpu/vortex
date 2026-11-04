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

// Functional test for VX_serial_mul and VX_serial_div as VX_alu_muldiv uses
// them at XLEN 64: the multiplier on 65-bit operands extended the way MUL,
// MULH, MULHSU and MULHU extend rs1/rs2 (the 128 low product bits and the
// sign), the divider against the RISC-V DIV/DIVU/REM/REMU definitions,
// including division by zero, the signed overflow and operands whose bit 31
// differs from their sign bit. Random and edge operands; one operation at a
// time through strobe/busy, both lanes at once; the step count is reported.

#include "VVX_serial_muldiv_top.h"
#include "vl_simulator.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

typedef unsigned __int128 u128;
typedef __int128 s128;

// bit fields of Verilator's wide ports (lane i of a packed array at [W*i +: W])
template <typename T> static void set_bits(T& w, int pos, int width, u128 v) {
    for (int b = 0; b < width; ++b, ++pos) {
        uint32_t bit = uint32_t((v >> b) & 1);
        w[pos / 32] = (w[pos / 32] & ~(1u << (pos % 32))) | (bit << (pos % 32));
    }
}
template <typename T> static u128 get_bits(const T& w, int pos, int width) {
    u128 v = 0;
    for (int b = 0; b < width; ++b, ++pos) v |= u128((w[pos / 32] >> (pos % 32)) & 1) << b;
    return v;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t n = 200000;
    if (argc > 1) n = std::strtoull(argv[1], nullptr, 0);

    vl_simulator<VVX_serial_muldiv_top> sim;
    std::mt19937_64 rng(0x5E41A1u);
    const uint64_t edges[] = {0, 1, 2, 3, 0x7fffffffull, 0x80000000ull, 0xffffffffull, 0x100000000ull,
                              0x7fffffffffffffffull, 0x8000000000000000ull, ~0ull, ~1ull,
                              0x00000000ffffffffull, 0x0000000080000000ull, 0xffffffff80000000ull,
                              0x80000000ffffffffull, 0x123456789abcdef0ull};
    const int ne = sizeof(edges) / sizeof(edges[0]);
    auto operand = [&](size_t i) -> uint64_t {
        switch (i % 5) {
        case 0: return edges[rng() % ne];
        case 1: return rng() & ((1ull << (rng() % 64)) - 1);  // small magnitudes
        default: return rng();
        }
    };

    sim->mul_strobe = 0; sim->div_strobe = 0; sim->div_signed = 0;
    uint64_t ticks = sim.reset(0);

    size_t errors = 0, muls = 0, divs = 0, mul_steps = 0, div_steps = 0;
    for (size_t i = 0; i < n; ++i) {
        // ---- multiply: 0 MUL/MULH (signed x signed), 1 MULHSU, 2 MULHU
        {
            int kind = int(rng() % 3);
            uint64_t a[2], b[2];
            s128 ax[2], bx[2];
            for (int l = 0; l < 2; ++l) {
                a[l] = operand(i + l); b[l] = operand(i + 2 + l);
                ax[l] = (kind == 2) ? s128(a[l]) : s128(int64_t(a[l]));
                bx[l] = (kind != 0) ? s128(b[l]) : s128(int64_t(b[l]));
                set_bits(sim->mul_a, 65 * l, 65, u128(ax[l]));
                set_bits(sim->mul_b, 65 * l, 65, u128(bx[l]));
            }
            sim->mul_strobe = 1;
            ticks = sim.step(ticks, 2);
            sim->mul_strobe = 0;
            size_t steps = 0;
            while (sim->mul_busy) { ticks = sim.step(ticks, 2); ++steps; if (steps > 200) break; }
            mul_steps = steps;
            for (int l = 0; l < 2; ++l) {
                u128 p = u128(ax[l]) * u128(bx[l]);  // the product mod 2^128
                bool neg = (ax[l] < 0) != (bx[l] < 0) && ax[l] != 0 && bx[l] != 0;
                u128 got = get_bits(sim->mul_r, 130 * l, 128);
                int sign = int(get_bits(sim->mul_r, 130 * l + 129, 1));
                if (got != p || sign != int(neg)) {
                    if (errors < 16)
                        printf("MISMATCH mul kind=%d lane=%d a=%016llx b=%016llx: got %016llx%016llx s%d exp %016llx%016llx s%d\n",
                               kind, l, (unsigned long long)a[l], (unsigned long long)b[l],
                               (unsigned long long)(got >> 64), (unsigned long long)got, sign,
                               (unsigned long long)(p >> 64), (unsigned long long)p, int(neg));
                    ++errors;
                }
            }
            ++muls;
        }
        // ---- divide
        {
            bool sgn = rng() & 1;
            uint64_t nn[2], dd[2];
            for (int l = 0; l < 2; ++l) {
                nn[l] = operand(i + 1 + l);
                dd[l] = (i % 7 == 3) ? 0 : (i % 11 == 5 ? ~0ull : operand(i + 3 + l));
                if (i % 13 == 6) { nn[l] = 0x8000000000000000ull; dd[l] = ~0ull; }  // signed overflow
            }
            for (int l = 0; l < 2; ++l) {
                set_bits(sim->div_n, 64 * l, 64, nn[l]);
                set_bits(sim->div_d, 64 * l, 64, dd[l]);
            }
            sim->div_signed = sgn;
            sim->div_strobe = 1;
            ticks = sim.step(ticks, 2);
            sim->div_strobe = 0;
            size_t steps = 0;
            while (sim->div_busy) { ticks = sim.step(ticks, 2); ++steps; if (steps > 200) break; }
            div_steps = steps;
            for (int l = 0; l < 2; ++l) {
                uint64_t q, r;
                if (dd[l] == 0) { q = ~0ull; r = nn[l]; }
                else if (sgn) {
                    int64_t a = int64_t(nn[l]), b = int64_t(dd[l]);
                    if (a == INT64_MIN && b == -1) { q = uint64_t(a); r = 0; }
                    else { q = uint64_t(a / b); r = uint64_t(a % b); }
                } else { q = nn[l] / dd[l]; r = nn[l] % dd[l]; }
                uint64_t gq = uint64_t(get_bits(sim->div_q, 64 * l, 64));
                uint64_t gr = uint64_t(get_bits(sim->div_r, 64 * l, 64));
                if (gq != q || gr != r) {
                    if (errors < 16)
                        printf("MISMATCH div s=%d lane=%d n=%016llx d=%016llx: got q=%016llx r=%016llx exp q=%016llx r=%016llx\n",
                               int(sgn), l, (unsigned long long)nn[l], (unsigned long long)dd[l],
                               (unsigned long long)gq, (unsigned long long)gr, (unsigned long long)q, (unsigned long long)r);
                    ++errors;
                }
            }
            ++divs;
        }
    }
    printf("checked=%zu muls, %zu divs (x2 lanes); mul busy %zu cycles, div busy %zu cycles; errors=%zu\n",
           muls, divs, mul_steps, div_steps, errors);
    printf("%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
