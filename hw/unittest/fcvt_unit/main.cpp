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

// Functional test for VX_fcvt_unit: exercises I2F and F2I in both signed and
// unsigned forms, plus F32<->F16 with HALF; at XLEN=64 also the I64 forms, and
// at FLEN=64 the F64 conversions (I32/I64 <-> F64, F32 <-> F64, F64 <-> F16).
// Checks bit-exact result/fflags against the softfloat (rvfloats) RISC-V golden
// across all rounding modes. A 32-bit integer result is checked in its low 32
// bits; a float result in its NaN-boxed XLEN bits. Strict check.

#include "VVX_fcvt_unit.h"
#include "vl_simulator.h"
#include "rvfloats.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#ifndef LATENCY
#define LATENCY 5
#endif
#ifndef FLEN
#define FLEN 32
#endif

using Device = VVX_fcvt_unit;

namespace {

constexpr bool kX64 = (VX_CFG_XLEN == 64);
constexpr bool kD   = (FLEN == 64);
#ifdef HALF
constexpr bool kH   = true;
#else
constexpr bool kH   = false;
#endif

enum Fmt { I32, U32, I64, U64, F32, F64, F16 };

struct Op {
    const char* name;
    Fmt src, dst;
    bool enabled;
};

const Op kOps[] = {
    {"i2f_s", I32, F32, true}, {"i2f_u", U32, F32, true},
    {"f2i_s", F32, I32, true}, {"f2i_u", F32, U32, true},
    {"f2h",   F32, F16, kH},   {"h2f",   F16, F32, kH},
    {"l2f_s", I64, F32, kX64}, {"l2f_u", U64, F32, kX64},
    {"f2l_s", F32, I64, kX64}, {"f2l_u", F32, U64, kX64},
    {"i2d_s", I32, F64, kD},   {"i2d_u", U32, F64, kD},
    {"d2i_s", F64, I32, kD},   {"d2i_u", F64, U32, kD},
    {"l2d_s", I64, F64, kD && kX64}, {"l2d_u", U64, F64, kD && kX64},
    {"d2l_s", F64, I64, kD && kX64}, {"d2l_u", F64, U64, kD && kX64},
    {"f2d",   F32, F64, kD},   {"d2f",   F64, F32, kD},
    {"d2h",   F64, F16, kD && kH}, {"h2d", F16, F64, kD && kH},
};
constexpr int NUM_OPS = sizeof(kOps) / sizeof(kOps[0]);

bool is_int(Fmt f) { return f <= U64; }

struct TV { uint64_t a; uint32_t frm; int op; };
struct Golden { uint64_t res, mask; uint32_t fflags; uint64_t a; uint32_t frm; int op; };

// Mix of float specials (for F2I) and integer specials (for I2F); a 32-bit word
// is reinterpreted per op.
const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x4f000000, 0xcf000000, 0x4effffff,
    0x00000001, 0xffffffff, 0x7fffffff, 0x00000002, 0x000003e8, 0x40490fdb,
    0x5f000000, 0x5effffff, 0x3f000000, 0x4c3b0244,
    0x477fe000, 0x477fefff, 0x477ff000, 0x477ff001, 0x47800000, 0xc77ff000,
    0x38800000, 0x387fffff, 0x387ff000, 0x387fe000, 0x33800000, 0x33000000,
    0x33000001, 0x32ffffff, 0x3f801000, 0x3f803000, 0x7f800001, 0x7f7fffff,
    0xffff7c00, 0xffff7c01, 0xffff7e00, 0xffff0001, 0xffff83ff, 0xffff3c00,
    0x00003c00, 0x7fff7bff,
    // I64 range edges as F32: +-2^63, +-2^64, just below, and beyond
    0x5f800000, 0xdf000000, 0xdf800000, 0xdeffffff, 0x5f7fffff, 0xe0000000,
};

// 64-bit specials: I64 edges (for I2F) and F64 values (for F2I / F2F).
const uint64_t kSpecials64[] = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x7fffffffffffffffull, 0xffffffffffffffffull,
    0x0000000000000001ull, 0x8000000000000001ull, 0x00000000ffffffffull, 0xffffffff00000000ull,
    0x0020000000000001ull, 0x0020000000000003ull, 0x7ffffffffffffc00ull, 0x7ffffffffffffe00ull,
    0x3ff0000000000000ull, 0xbff0000000000000ull, 0x43e0000000000000ull, 0xc3e0000000000000ull,
    0x43efffffffffffffull, 0xc3efffffffffffffull, 0x43f0000000000000ull, 0xc3f0000000000000ull,
    0x41dfffffffc00000ull, 0x41e0000000000000ull, 0xc1e0000000000000ull, 0xc1e0000000200000ull,
    0x41efffffffe00000ull, 0x41effffffff00000ull, 0x41f0000000000000ull, 0x3fe0000000000000ull,
    0x3fe0000000000001ull, 0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull,
    0x7ff0000000000001ull, 0x0000000000000001ull, 0x000fffffffffffffull, 0x0010000000000000ull,
    0x47efffffe0000000ull, 0x47efffffefffffffull, 0x47effffff0000000ull, 0x3810000000000000ull,
    0x380fffffffffffffull, 0x36a0000000000000ull, 0x369fffffffffffffull, 0x36a0000000000001ull,
    0x40effe0000000000ull, 0x40effc0000000000ull, 0x3f10000000000000ull, 0x3e70000000000000ull,
};

inline uint32_t flush_sub(uint32_t x){ return (((x>>23)&0xFF)==0 && (x&0x7FFFFF)) ? (x&0x80000000u) : x; }
inline uint16_t flush_sub16(uint16_t x){ return (((x>>10)&0x1F)==0 && (x&0x3FF)) ? (x&0x8000u) : x; }
inline uint64_t flush_sub64(uint64_t x){ return (((x>>52)&0x7FF)==0 && (x&0xFFFFFFFFFFFFFull)) ? (x&0x8000000000000000ull) : x; }

uint64_t box32(uint32_t x) { return kX64 ? (0xffffffff00000000ull | x) : x; }
uint64_t box16(uint16_t x) { return (kX64 ? 0xffffffffffff0000ull : 0xffff0000ull) | x; }

// The source operand register as the core presents it.
uint64_t source(Fmt f, uint64_t r) {
    switch (f) {
    case F32: return box32(uint32_t(r));
    case F16: return (r >> 16 & 1) ? box16(uint16_t(r)) : uint32_t(r); // some unboxed halves
    case I32: case U32: return uint32_t(r);
    default:  return r;
    }
}

uint64_t golden(const Op& op, uint64_t a, uint32_t frm, uint32_t* ff) {
    *ff = 0;
    const uint32_t a32 = uint32_t(a);
    const bool h_boxed = kX64 ? ((a >> 16) == 0xffffffffffffull) : ((a32 >> 16) == 0xffff);
    const bool s_boxed = !kD || !kX64 || ((a >> 32) == 0xffffffffull);
    switch (op.src) {
    case I32: return op.dst == F32 ? box32(rv_itof_s(a32, frm, ff)) : rv_itof_d(a32, frm, ff);
    case U32: return op.dst == F32 ? box32(rv_utof_s(a32, frm, ff)) : rv_utof_d(a32, frm, ff);
    case I64: return op.dst == F32 ? box32(rv_ltof_s(a, frm, ff))   : rv_ltof_d(a, frm, ff);
    case U64: return op.dst == F32 ? box32(rv_lutof_s(a, frm, ff))  : rv_lutof_d(a, frm, ff);
    case F32: {
        const uint32_t s = s_boxed ? a32 : 0x7fc00000u;
        switch (op.dst) {
        case I32: return rv_ftoi_s(s, frm, ff);
        case U32: return rv_ftou_s(s, frm, ff);
        case I64: return rv_ftol_s(s, frm, ff);
        case U64: return rv_ftolu_s(s, frm, ff);
        case F16: return box16(rv_ftoh_s(s, frm, ff));
        default:  return rv_ftod(s, frm, ff);
        }
    }
    case F64:
        switch (op.dst) {
        case I32: return rv_ftoi_d(a, frm, ff);
        case U32: return rv_ftou_d(a, frm, ff);
        case I64: return rv_ftol_d(a, frm, ff);
        case U64: return rv_ftolu_d(a, frm, ff);
        case F16: return box16(rv_dtoh(a, frm, ff));
        default:  return box32(rv_dtof(a, frm, ff));
        }
    case F16:
        if (op.dst == F32) return box32(h_boxed ? rv_htof_s(uint16_t(a), frm, ff) : 0x7fc00000u);
        return h_boxed ? rv_htod(uint16_t(a), frm, ff) : 0x7ff8000000000000ull;
    }
    return 0;
}

void drive(Device& dut, const Op& op, uint64_t a, uint32_t frm) {
    const bool f2f = !is_int(op.src) && !is_int(op.dst);
    const Fmt  ifmt = is_int(op.src) ? op.src : op.dst;
    dut.frm       = frm;
    dut.is_itof   = is_int(op.src);
    dut.is_ftoi   = is_int(op.dst);
    dut.is_f2f    = f2f;
    dut.is_signed = (ifmt == I32 || ifmt == I64);
    dut.is_int64  = !f2f && (ifmt == I64 || ifmt == U64);
    dut.src_fmt   = (op.src == F64) || (is_int(op.src) && op.dst == F64);
    dut.dst_fmt   = (op.dst == F64) || (is_int(op.dst) && op.src == F64);
#ifdef HALF
    dut.src_h     = (op.src == F16);
    dut.dst_h     = (op.dst == F16);
#endif
    dut.dataa     = a;
    dut.mask      = 1;
}

uint64_t result_mask(const Op& op) {
    if (op.dst == I32 || op.dst == U32) return 0xffffffffull;
    return kX64 ? ~0ull : 0xffffffffull;
}

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    size_t num_random = 100000; // per op
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_random = static_cast<size_t>(v);
    }

    std::vector<TV> tests;
    std::mt19937_64 rng(0xC0117EEu);
    for (int o = 0; o < NUM_OPS; ++o) {
        const Op& op = kOps[o];
        if (!op.enabled) continue;
        const bool wide = (op.src == I64 || op.src == U64 || op.src == F64);
        for (uint32_t f = 0; f < 5; ++f) {
            for (uint32_t a : kSpecials)
                tests.push_back({source(op.src, a), f, o});
            if (wide) {
                for (uint64_t a : kSpecials64)
                    tests.push_back({a, f, o});
            }
        }
        for (size_t i = 0; i < num_random; ++i) {
            uint64_t r = rng();
            if (op.src == F16 && (i % 16) != 0) {
                r |= 0xffff0000u;
            }
            if (op.src == F32 && op.dst == F16 && (i % 2) == 0) {
                r = (r & 0x807fffffu) | ((97u + (uint32_t(r) >> 24) % 50u) << 23);
            }
            if (op.src == F64 && (i % 2) == 0) {
                // exponents around the I32/I64 and F32/F16 range edges
                const uint64_t e = 1023 - 30 + (r >> 52) % 100;
                r = (r & 0x800fffffffffffffull) | (e << 52);
            }
            if ((op.src == I64 || op.src == U64) && (i % 4) == 0) {
                r >>= (r & 63); // short magnitudes
            }
            if (op.src == F32 && is_int(op.dst) && (i % 2) == 0) {
                const uint32_t e = 127 - 2 + uint32_t(r >> 32) % 70u; // around the I32/I64 edges
                r = (r & 0x807fffffu) | (e << 23);
            }
            tests.push_back({source(op.src, r), static_cast<uint32_t>(i % 5), o});
        }
    }

    const size_t N = tests.size();

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1; sim->mask = 0;
    sim->frm = 0; sim->is_itof = 0; sim->is_ftoi = 0; sim->is_f2f = 0;
    sim->is_signed = 0; sim->is_int64 = 0; sim->src_fmt = 0; sim->dst_fmt = 0;
    sim->dataa = 0;
    ticks = sim.reset(ticks);

    std::deque<Golden> fifo;
    size_t feed = 0, checked = 0, errors = 0, per_op[NUM_OPS] = {0};

    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            const TV& t = tests[feed];
            const Op& op = kOps[t.op];
            drive(*sim.operator->(), op, t.a, t.frm);
            uint32_t ff = 0;
            uint64_t ga = t.a;
#ifdef LEAN
            if (op.src == F32) {
                ga = box32(flush_sub(uint32_t(t.a))); // DAZ float source
            }
            if (op.src == F64) {
                ga = flush_sub64(t.a);
            }
            if (op.src == F16 && (uint32_t(t.a) >> 16) == 0xffff) {
                ga = box16(flush_sub16(uint16_t(t.a)));
            }
#endif
            uint64_t r = golden(op, ga, t.frm, &ff);
#ifdef LEAN
            ff = 0; // EXCEPT_ENABLE=0 -> no fflags
#endif
            fifo.push_back({r, result_mask(op), ff, t.a, t.frm, t.op});
            ++feed;
        } else {
            sim->mask = 0;
        }
        sim->enable = 1;
        ticks = sim.step(ticks, 2);

        if (c >= (size_t)(LATENCY - 1) && !fifo.empty()) {
            Golden g = fifo.front(); fifo.pop_front();
            uint64_t got_r = uint64_t(sim->result) & g.mask;
            uint32_t got_f = sim->fflags;
            if (got_r != (g.res & g.mask) || got_f != g.fflags) {
                ++per_op[g.op];
                if (errors < 24)
                    printf("MISMATCH %-6s a=%016llx frm=%u : got res=%016llx ff=%02x  exp res=%016llx ff=%02x\n",
                           kOps[g.op].name, (unsigned long long)g.a, g.frm, (unsigned long long)got_r, got_f,
                           (unsigned long long)(g.res & g.mask), g.fflags);
                ++errors;
            }
            ++checked;
        }
    }

    printf("fcvt_unit: checked=%zu errors=%zu  per-op:", checked, errors);
    for (int o = 0; o < NUM_OPS; ++o) {
        if (kOps[o].enabled) printf(" %s=%zu", kOps[o].name, per_op[o]);
    }
    printf("\n%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
