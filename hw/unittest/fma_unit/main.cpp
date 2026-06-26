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

// Functional test for VX_fma_unit (F32): exercises the whole FMA op family
// (MUL/ADD/SUB/MADD/MSUB/NMADD/NMSUB) and checks bit-exact result/fflags against
// the softfloat (rvfloats) RISC-V golden across all rounding modes. The unit
// remaps ADD/SUB/MUL onto its a*b+/-c datapath internally; this test drives the
// same input convention VX_fpu_dsp/VX_fpu_std use. Strict check; per-op buckets.

#include "VVX_fma_unit.h"
#include "vl_simulator.h"
#include "rvfloats.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

#ifndef LATENCY
#define LATENCY 6
#endif

using Device = VVX_fma_unit;

// op_type encodings (VX_gpu_pkg)
enum { OP_ADD = 0, OP_MUL = 1, OP_MADD = 2, OP_NMADD = 3 };
// op index for the test
enum { FADD, FSUB, FMUL, FMADD, FMSUB, FNMADD, FNMSUB, NUM_OPS };

namespace {

const char* op_name[NUM_OPS] = {"fadd","fsub","fmul","fmadd","fmsub","fnmadd","fnmsub"};

struct TV { uint32_t a, b, c; uint32_t frm; int op; };
struct Golden { uint32_t res, fflags, a, b, c, frm; int op; };

const uint32_t kSpecials[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
    0x00000001, 0x807fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
    0x40490fdb, 0x3dcccccd, 0x41200000,
};

// A "tame" F32 with exponent near 127 so products stay in normal range.
uint32_t tame(std::mt19937& rng) {
    uint32_t sign = (rng() & 1u) << 31;
    uint32_t exp  = 110 + (rng() % 30);   // [110,139]
    uint32_t man  = rng() & 0x7FFFFF;
    return sign | (exp << 23) | man;
}

// Map (op, a, b, c) to RTL op_type/fmt and the rvfloats golden.
void encode(int op, uint32_t a, uint32_t b, uint32_t c,
            uint32_t& op_type, uint32_t& fmt,
            uint32_t& da, uint32_t& db, uint32_t& dc) {
    fmt = 0; da = a; db = b; dc = c;
    switch (op) {
    case FADD:   op_type = OP_ADD;   fmt = 0; db = b; dc = 0; break; // a*1 + b (b->op_c)
    case FSUB:   op_type = OP_ADD;   fmt = 2; db = b; dc = 0; break; // a*1 - b
    case FMUL:   op_type = OP_MUL;   fmt = 0; dc = 0;         break; // a*b + 0
    case FMADD:  op_type = OP_MADD;  fmt = 0;                 break; // a*b + c
    case FMSUB:  op_type = OP_MADD;  fmt = 2;                 break; // a*b - c
    case FNMADD: op_type = OP_NMADD; fmt = 0;                 break; // -(a*b) - c
    case FNMSUB: op_type = OP_NMADD; fmt = 2;                 break; // -(a*b) + c
    }
}

uint32_t golden(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* ff) {
    *ff = 0;
    switch (op) {
    case FADD:   return rv_fadd_s(a, b, frm, ff);
    case FSUB:   return rv_fsub_s(a, b, frm, ff);
    case FMUL:   return rv_fmul_s(a, b, frm, ff);
    case FMADD:  return rv_fmadd_s(a, b, c, frm, ff);
    case FMSUB:  return rv_fmsub_s(a, b, c, frm, ff);
    case FNMADD: return rv_fnmadd_s(a, b, c, frm, ff);
    case FNMSUB: return rv_fnmsub_s(a, b, c, frm, ff);
    }
    return 0;
}

inline int fclass_ns(uint32_t x){ uint32_t e=(x>>23)&0xFF,m=x&0x7FFFFF; if(e==0)return m?1:0; if(e==0xFF)return m?4:3; return 2; }
inline uint32_t flush_sub(uint32_t x){ return (((x>>23)&0xFF)==0 && (x&0x7FFFFF)) ? (x&0x80000000u) : x; }

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    size_t num_random = 60000; // per op
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) num_random = static_cast<size_t>(v);
    }

    std::vector<TV> tests;
    std::mt19937 rng(0xFAACEEu);
    for (int op = 0; op < NUM_OPS; ++op) {
        // directed specials
        for (uint32_t a : kSpecials)
            for (uint32_t b : kSpecials)
                for (uint32_t f = 0; f < 5; ++f)
                    tests.push_back({a, b, 0x3f000000u, f, op});
        // tame random (normal-range coverage)
        for (size_t i = 0; i < num_random; ++i)
            tests.push_back({tame(rng), tame(rng), tame(rng),
                             static_cast<uint32_t>(i % 5), op});
        // full random
        for (size_t i = 0; i < num_random; ++i)
            tests.push_back({static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()),
                             static_cast<uint32_t>(rng()), static_cast<uint32_t>(i % 5), op});
    }

#ifdef LEAN
    // both SNORM/EXCEPT disabled -> normal operands only (RTU domain); FTZ + no fflags.
    { std::vector<TV> f; for (auto& t : tests) {
        bool ok = fclass_ns(t.a)==2 && fclass_ns(t.b)==2 && (t.op < FMADD || fclass_ns(t.c)==2);
        if (!ok) continue;
        uint32_t ff=0; (void)golden(t.op, t.a, t.b, t.c, t.frm, &ff);
        if (ff & 0x18) continue;
        f.push_back(t);
    } tests.swap(f); }
#endif

    const size_t N = tests.size();

    vl_simulator<Device> sim;
    uint64_t ticks = 0;
    sim->enable = 1; sim->mask = 0; sim->op_type = 0; sim->fmt = 0; sim->frm = 0;
    sim->dataa = 0; sim->datab = 0; sim->datac = 0;
    ticks = sim.reset(ticks);

    auto fclass = [](uint32_t x) { // 0=zero 1=sub 2=norm 3=inf 4=nan
        uint32_t e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
        if (e == 0) return m ? 1 : 0;
        if (e == 0xFF) return m ? 4 : 3;
        return 2;
    };

    std::deque<Golden> fifo;
    size_t feed = 0, checked = 0, errors = 0;
    size_t per_op[NUM_OPS] = {0};
    size_t e_ovf = 0, e_uf = 0, e_special = 0, e_normal = 0;

    for (size_t c = 0; c < N + LATENCY; ++c) {
        if (feed < N) {
            const TV& t = tests[feed];
            uint32_t op_type, fmt, da, db, dc;
            encode(t.op, t.a, t.b, t.c, op_type, fmt, da, db, dc);
            sim->op_type = op_type; sim->fmt = fmt; sim->frm = t.frm;
            sim->dataa = da; sim->datab = db; sim->datac = dc; sim->mask = 1;
            uint32_t ff = 0;
            uint32_t r  = golden(t.op, t.a, t.b, t.c, t.frm, &ff);
#ifdef LEAN
            r = flush_sub(r); ff = 0;
#endif
            fifo.push_back({r, ff, t.a, t.b, t.c, t.frm, t.op});
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
                ++per_op[g.op];
                bool all_norm = fclass(g.a) == 2 && fclass(g.b) == 2 &&
                                (g.op < FMADD || fclass(g.c) == 2);
                bool norm_bucket = !(g.fflags & 0x6) && all_norm;
                if (g.fflags & 0x4)      ++e_ovf;
                else if (g.fflags & 0x2) ++e_uf;
                else if (all_norm)       ++e_normal;
                else ++e_special;
                if (errors < 16)
                    printf("MISMATCH %-6s a=%08x b=%08x c=%08x frm=%u : got res=%08x ff=%02x  exp res=%08x ff=%02x %s\n",
                           op_name[g.op], g.a, g.b, g.c, g.frm, got_r, got_f, g.res, g.fflags,
                           norm_bucket ? "[normal]" : "");
                ++errors;
            }
            ++checked;
        }
    }

    printf("fma_unit: checked=%zu errors=%zu  [overflow=%zu underflow=%zu special-operand=%zu normal=%zu]\n",
           checked, errors, e_ovf, e_uf, e_special, e_normal);
    printf("  per-op:");
    for (int op = 0; op < NUM_OPS; ++op) printf(" %s=%zu", op_name[op], per_op[op]);
    printf("\n%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
