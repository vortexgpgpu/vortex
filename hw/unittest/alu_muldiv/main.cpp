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

// Functional and timing test of VX_alu_muldiv as the core builds it for
// synthesis (no DPI models): at XLEN 64 the multi-pass multiplier, at XLEN 32
// the pipelined one, and the iterative divider. Every M opcode (with and
// without W at XLEN 64) against the RISC-V definitions (the double-width
// products of MULH/MULHSU/MULHU, division by zero and the signed overflow),
// on random and edge operands per lane.
//
// Phase 1 (random): random opcode mix, the input valid and the output ready
// toggled at random, so operations are held at the input mid-pass and the
// pipeline stalls with results in flight; every result is checked by tag.
// Phase 2 (timing): back-to-back streams of one multiply kind with the
// output always ready; reports cycles per operation and the latency, which
// must be 1/3/4 cycles and LATENCY_IMUL + passes + 1 for W/low/high at XLEN
// 64, and 1 cycle and LATENCY_IMUL + 1 at XLEN 32 (the response buffer adds
// the + 1 in both).

#include "VVX_alu_muldiv_top.h"
#include "vl_simulator.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <vector>

typedef unsigned __int128 u128;
typedef __int128 s128;

static const int LANES = VX_CFG_NUM_ALU_LANES;
static const int XL = VX_CFG_XLEN;
static const uint64_t XMASK = (XL == 64) ? ~0ull : 0xffffffffull;

enum { M_MUL = 0, M_MULHU = 1, M_MULH = 2, M_MULHSU = 3, M_DIV = 4, M_DIVU = 5, M_REM = 6, M_REMU = 7 };

// lane i of a packed [LANES][XLEN] port
template <typename T> static void set_x(T& w, int lane, uint64_t v) {
    if (XL == 64) {
        w[2 * lane] = uint32_t(v);
        w[2 * lane + 1] = uint32_t(v >> 32);
    } else {
        w[lane] = uint32_t(v);
    }
}
template <typename T> static uint64_t get_x(const T& w, int lane) {
    if (XL == 64)
        return uint64_t(w[2 * lane]) | (uint64_t(w[2 * lane + 1]) << 32);
    return w[lane];
}

static uint64_t sext32(uint64_t x) { return uint64_t(int64_t(int32_t(uint32_t(x)))); }

// the 32-bit forms: RV64's W operations, or XLEN 32's divides
static uint64_t ref_w(int op, uint64_t a, uint64_t b);

static uint64_t ref(int op, bool w, uint64_t a, uint64_t b) {
    if (XL == 32) {
        int64_t sa = int32_t(uint32_t(a)), sb = int32_t(uint32_t(b));
        uint64_t ua = uint32_t(a), ub = uint32_t(b);
        switch (op) {
        case M_MUL:    return uint32_t(ua * ub);
        case M_MULH:   return uint32_t(uint64_t(sa * sb) >> 32);
        case M_MULHSU: return uint32_t(uint64_t(sa * int64_t(ub)) >> 32);
        case M_MULHU:  return uint32_t((ua * ub) >> 32);
        default:       return uint32_t(ref_w(op, a, b));
        }
    }
    if (w)
        return ref_w(op, a, b);
    int64_t sa = int64_t(a), sb = int64_t(b);
    switch (op) {
    case M_MUL:    return a * b;
    case M_MULH:   return uint64_t(u128(s128(sa) * s128(sb)) >> 64);
    case M_MULHSU: return uint64_t(u128(s128(sa) * s128(u128(b))) >> 64);
    case M_MULHU:  return uint64_t((u128(a) * u128(b)) >> 64);
    case M_DIV:    return b == 0 ? ~0ull : (sa == INT64_MIN && sb == -1) ? a : uint64_t(sa / sb);
    case M_DIVU:   return b == 0 ? ~0ull : a / b;
    case M_REM:    return b == 0 ? a : (sa == INT64_MIN && sb == -1) ? 0 : uint64_t(sa % sb);
    case M_REMU:   return b == 0 ? a : a % b;
    default: std::abort();
    }
}

static uint64_t ref_w(int op, uint64_t a, uint64_t b) {
    int32_t sa = int32_t(uint32_t(a)), sb = int32_t(uint32_t(b));
    uint32_t ua = uint32_t(a), ub = uint32_t(b);
    switch (op) {
    case M_MUL:  return sext32(uint64_t(ua * ub));
    case M_DIV:  return sext32(ub == 0 ? ~0u : (sa == INT32_MIN && sb == -1) ? uint32_t(sa) : uint32_t(sa / sb));
    case M_DIVU: return sext32(ub == 0 ? ~0u : ua / ub);
    case M_REM:  return sext32(ub == 0 ? ua : (sa == INT32_MIN && sb == -1) ? 0u : uint32_t(sa % sb));
    case M_REMU: return sext32(ub == 0 ? ua : ua % ub);
    default: std::abort();
    }
}

struct Op {
    int op;
    bool w;
    uint32_t tag;
    uint64_t a[LANES], b[LANES];
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t n = 20000;
    if (argc > 1) n = std::strtoull(argv[1], nullptr, 0);

    vl_simulator<VVX_alu_muldiv_top> sim;
    std::mt19937_64 rng(0xA1D1Fu);
    const uint64_t edges[] = {0, 1, 2, 3, 0x7fffffffull, 0x80000000ull, 0xffffffffull, 0x100000000ull,
                              0x7fffffffffffffffull, 0x8000000000000000ull, ~0ull, ~1ull,
                              0xffffffff80000000ull, 0x80000000ffffffffull, 0x00000001ffffffffull,
                              0x123456789abcdef0ull, 0xfedcba9876543210ull};
    const int ne = sizeof(edges) / sizeof(edges[0]);
    auto operand = [&]() -> uint64_t {
        switch (rng() % 6) {
        case 0: case 1: return edges[rng() % ne];
        case 2: return rng() & ((1ull << (rng() % 64)) - 1);  // small magnitudes
        case 3: return ~(rng() & ((1ull << (rng() % 64)) - 1)); // small negatives
        default: return rng();
        }
    };
    auto make_op = [&](int op, bool w, uint32_t tag) {
        Op o; o.op = op; o.w = w; o.tag = tag;
        for (int l = 0; l < LANES; ++l) { o.a[l] = operand() & XMASK; o.b[l] = operand() & XMASK; }
        return o;
    };
    auto drive = [&](const Op& o) {
        sim->in_op = o.op; sim->in_is_w = o.w; sim->in_tag = o.tag;
        for (int l = 0; l < LANES; ++l) { set_x(sim->in_rs1, l, o.a[l]); set_x(sim->in_rs2, l, o.b[l]); }
    };

    sim->in_valid = 0; sim->out_ready = 0;
    uint64_t ticks = sim.reset(0);

    size_t errors = 0;
    std::map<uint32_t, Op> pending;
    auto check = [&]() {
        if (!(sim->out_valid && sim->out_ready)) return;
        uint32_t tag = sim->out_tag;
        auto it = pending.find(tag);
        if (it == pending.end()) {
            if (errors++ < 16) printf("UNEXPECTED result tag %u\n", tag);
            return;
        }
        const Op& o = it->second;
        for (int l = 0; l < LANES; ++l) {
            uint64_t exp = ref(o.op, o.w, o.a[l], o.b[l]);
            uint64_t got = get_x(sim->out_data, l);
            if (got != exp && errors++ < 16)
                printf("MISMATCH op=%d w=%d lane=%d a=%016llx b=%016llx: got %016llx exp %016llx\n", o.op, int(o.w), l,
                       (unsigned long long)o.a[l], (unsigned long long)o.b[l], (unsigned long long)got, (unsigned long long)exp);
        }
        pending.erase(it);
    };

    // ---- phase 1: random mix with input bubbles and output stalls
    {
        size_t issued = 0, cycles = 0;
        bool have = false; Op cur;
        while (issued < n || !pending.empty() || have) {
            if (!have && issued < n && (rng() % 4) != 0) {
                int op = int(rng() % 8);
                // DIV is 64x slower: keep about one divide per eight operations
                if (op >= M_DIV && (rng() % 4) != 0) op = int(rng() % 4);
                bool w = (XL == 64) && (op == M_MUL || op >= M_DIV) && (rng() & 1);
                cur = make_op(op, w, uint32_t(issued));
                have = true; ++issued;
            }
            sim->in_valid = have;
            if (have) drive(cur);
            sim->out_ready = (rng() % 3) != 0;
            sim->eval();
            check();
            bool fire = have && sim->in_ready;
            if (fire) { pending[cur.tag] = cur; have = false; }
            ticks = sim.step(ticks, 2);
            if (++cycles > 400 * n + 10000) { printf("TIMEOUT in phase 1: %zu pending\n", pending.size()); ++errors; break; }
        }
        printf("phase 1: %zu operations x %d lanes in %zu cycles\n", n, LANES, cycles);
    }

    // ---- phase 2: back-to-back streams of one multiply kind, output always ready
    struct Kind { const char* name; int op; bool w; int passes; };
    std::vector<Kind> kinds;
    if (XL == 64)
        kinds = {{"MULW", M_MUL, true, 1}, {"MUL", M_MUL, false, 3}, {"MULH", M_MULH, false, 4},
                 {"MULHSU", M_MULHSU, false, 4}, {"MULHU", M_MULHU, false, 4}};
    else
        kinds = {{"MUL", M_MUL, false, 1}, {"MULH", M_MULH, false, 1},
                 {"MULHSU", M_MULHSU, false, 1}, {"MULHU", M_MULHU, false, 1}};
    const int STREAM = 64;
    const int LAT = 3; // LATENCY_IMUL
    for (auto& k : kinds) {
        std::deque<Op> q;
        for (int i = 0; i < STREAM; ++i) q.push_back(make_op(k.op, k.w, 0x100000 + i));
        size_t cycles = 0, first_out = 0, last_out = 0, outs = 0;
        sim->out_ready = 1;
        while (outs < (size_t)STREAM) {
            sim->in_valid = !q.empty();
            if (!q.empty()) drive(q.front());
            sim->eval();
            if (sim->out_valid) { if (!outs) first_out = cycles; last_out = cycles; ++outs; }
            check();
            if (!q.empty() && sim->in_ready) { pending[q.front().tag] = q.front(); q.pop_front(); }
            ticks = sim.step(ticks, 2);
            if (++cycles > 100000) { printf("TIMEOUT in phase 2 (%s)\n", k.name); ++errors; break; }
        }
        double per_op = double(last_out - first_out) / (STREAM - 1);
        // the first operation enters at cycle 0; its result leaves after its
        // passes, the multiplier, the accumulate stage (XLEN 64) and the
        // unit's response buffer
        size_t exp_lat = size_t(k.passes - 1 + LAT + (XL == 64 ? 1 : 0) + 1);
        bool ok = (last_out - first_out) == size_t(k.passes) * (STREAM - 1) && first_out == exp_lat;
        printf("phase 2: %-6s %.2f cycles/op (expect %d), latency %zu (expect %zu)%s\n", k.name, per_op, k.passes,
               first_out, exp_lat, ok ? "" : "  <-- WRONG");
        if (!ok) ++errors;
    }
    if (!pending.empty()) { printf("%zu results missing\n", pending.size()); ++errors; }

    printf("checked=%zu+%zu operations x %d lanes (XLEN %d); errors=%zu\n", n, STREAM * kinds.size(), LANES, XL, errors);
    printf("%s\n", errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
