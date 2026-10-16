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

// Functional test for VX_fcvt_unit (XLEN=32 / F32): exercises I2F and F2I in
// both signed and unsigned forms, plus F32<->F16 with HALF, and checks
// bit-exact result/fflags against the softfloat (rvfloats) RISC-V golden across
// all rounding modes. Strict check.

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

using Device = VVX_fcvt_unit;

#ifdef HALF
enum { I2F_S, I2F_U, F2I_S, F2I_U, F2H, H2F, NUM_OPS };
#else
enum { I2F_S, I2F_U, F2I_S, F2I_U, NUM_OPS, F2H, H2F };
#endif

namespace {

const char* op_name[] = {"i2f_s","i2f_u","f2i_s","f2i_u","f2h","h2f"};

struct TV { uint32_t a, frm; int op; };
struct Golden { uint32_t res, fflags, a, frm; int op; };

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
};

inline uint32_t flush_sub(uint32_t x){ return (((x>>23)&0xFF)==0 && (x&0x7FFFFF)) ? (x&0x80000000u) : x; }
inline uint16_t flush_sub16(uint16_t x){ return (((x>>10)&0x1F)==0 && (x&0x3FF)) ? (x&0x8000u) : x; }

uint32_t golden(int op, uint32_t a, uint32_t frm, uint32_t* ff) {
    *ff = 0;
    switch (op) {
    case I2F_S: return rv_itof_s(a, frm, ff);
    case I2F_U: return rv_utof_s(a, frm, ff);
    case F2I_S: return rv_ftoi_s(a, frm, ff);
    case F2I_U: return rv_ftou_s(a, frm, ff);
    case F2H: return 0xffff0000u | rv_ftoh_s(a, frm, ff);
    case H2F: return ((a >> 16) == 0xffff) ? rv_htof_s(uint16_t(a), frm, ff) : 0x7fc00000u;
    }
    return 0;
}

void drive(Device& dut, int op, uint32_t a, uint32_t frm) {
    dut.frm      = frm;
    dut.is_itof  = (op == I2F_S || op == I2F_U);
    dut.is_ftoi  = (op == F2I_S || op == F2I_U);
    dut.is_f2f   = (op == F2H || op == H2F);
    dut.is_signed = (op == I2F_S || op == F2I_S);
    dut.is_int64 = 0;     // XLEN=32
    dut.src_fmt  = 0;     // F32
    dut.dst_fmt  = 0;     // F32
#ifdef HALF
    dut.src_h    = (op == H2F);
    dut.dst_h    = (op == F2H);
#endif
    dut.dataa    = a;
    dut.mask     = 1;
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
    std::mt19937 rng(0xC0117EEu);
    for (int op = 0; op < NUM_OPS; ++op) {
        for (uint32_t a : kSpecials)
            for (uint32_t f = 0; f < 5; ++f)
                tests.push_back({a, f, op});
        for (size_t i = 0; i < num_random; ++i) {
            uint32_t a = static_cast<uint32_t>(rng());
            if (op == H2F && (i % 16) != 0) a |= 0xffff0000u;
            if (op == F2H && (i % 2) == 0) a = (a & 0x807fffffu) | ((97u + (a >> 24) % 50u) << 23);
            tests.push_back({a, static_cast<uint32_t>(i % 5), op});
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
            drive(*sim.operator->(), t.op, t.a, t.frm);
            uint32_t ff = 0;
            uint32_t ga = t.a;
#ifdef LEAN
            if (t.op == F2I_S || t.op == F2I_U || t.op == F2H) ga = flush_sub(t.a); // DAZ float source
            if (t.op == H2F && (t.a >> 16) == 0xffff) ga = 0xffff0000u | flush_sub16(uint16_t(t.a));
#endif
            uint32_t r  = golden(t.op, ga, t.frm, &ff);
#ifdef LEAN
            ff = 0; // EXCEPT_ENABLE=0 -> no fflags
#endif
            fifo.push_back({r, ff, t.a, t.frm, t.op});
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
                if (errors < 24)
                    printf("MISMATCH %-6s a=%08x frm=%u : got res=%08x ff=%02x  exp res=%08x ff=%02x\n",
                           op_name[g.op], g.a, g.frm, got_r, got_f, g.res, g.fflags);
                ++errors;
            }
            ++checked;
        }
    }

    printf("fcvt_unit: checked=%zu errors=%zu  per-op:", checked, errors);
    for (int op = 0; op < (int)NUM_OPS; ++op) printf(" %s=%zu", op_name[op], per_op[op]);
    printf("\n%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
