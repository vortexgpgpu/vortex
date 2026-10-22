// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_launch_seq.
//
// The sequencer runs against:
//   - an AXI read slave over a byte memory, with random AR stalls and read
//     latency, that checks every read is one aligned 64-byte line;
//   - a DCR sink with random back-pressure and a DCR proxy that is busy for a
//     random time after the grant;
//   - a GPU whose busy rises and falls a random time after start.
// Each command's DCR writes, start and done/skip outcome are compared with a
// reference model of the launch descriptors. Covered: plain launch; QMD with
// 0/1/7/8/18/64 pairs and a corrupt count; indirect launch with and without a
// predicate (true, false, inverted, every word offset), with a record (grid
// substitution, constant-1 select, shifts, a zero in each dimension, records
// crossing a line, word 14), x bias and divider corners (0, 1, 2, 7,
// 2^32-1), the predicate-only form, misaligned descriptors (refused with
// `fault`), and a randomized sweep.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_launch_seq_top.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif
#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool     trace_en  = false;
double sc_time_stamp() { return timestamp; }
bool   sim_trace_enabled() { return trace_en; }
void   sim_trace_enable(bool e) { trace_en = e; }

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d [%s]: ", __FILE__, __LINE__, g_case); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr uint8_t  OP_LAUNCH     = 0x06;
constexpr uint8_t  OP_LAUNCH_QMD = 0x0B;
constexpr uint8_t  OP_LAUNCH_IND = 0x0D;
constexpr uint32_t DCR_GRID_X    = 0x019;
constexpr uint32_t HAS_RECORD    = 1u << 24;
constexpr uint32_t HAS_PRED      = 1u << 25;
constexpr uint32_t PRED_INV      = 1u << 26;
constexpr uint32_t NO_LAUNCH     = 1u << 27;
constexpr uint32_t SEL_ONE       = 0xF;
constexpr int      SETTLE        = 3;
constexpr size_t   MEM_SIZE      = 1u << 20;

const char*  g_case = "";
std::mt19937 g_rng(1234);

uint32_t rnd(uint32_t n) {
    return std::uniform_int_distribution<uint32_t>(0, n - 1)(g_rng);
}

std::vector<uint8_t> g_mem(MEM_SIZE, 0);

void wr32(uint64_t a, uint32_t v) { std::memcpy(&g_mem[a], &v, 4); }
void wr64(uint64_t a, uint64_t v) { std::memcpy(&g_mem[a], &v, 8); }
uint32_t rd32(uint64_t a) { uint32_t v; std::memcpy(&v, &g_mem[a], 4); return v; }

struct Pair { uint32_t addr, value; };

struct Ild {
    uint32_t ctl = 0;
    uint16_t skip_n = 0;
    uint64_t rec = 0, pred = 0;
    uint32_t ref = 0, bias = 0, div = 0;
    std::vector<Pair> pairs;
    uint32_t count_override = ~0u;   // corrupt count in the header
};

uint32_t ctl_sel(uint32_t x, uint32_t y, uint32_t z) {
    return x | (y << 4) | (z << 8);
}

uint32_t ctl_sh(uint32_t x, uint32_t y, uint32_t z) {
    return (x << 12) | (y << 16) | (z << 20);
}

void put_ild(uint64_t at, const Ild& d) {
    std::memset(&g_mem[at], 0, 64);
    wr32(at + 0, d.ctl);
    wr32(at + 4, d.skip_n);
    wr64(at + 8, d.rec);
    wr64(at + 16, d.pred);
    wr32(at + 24, d.ref);
    wr32(at + 28, d.count_override != ~0u ? d.count_override : uint32_t(d.pairs.size()));
    wr32(at + 32, d.bias);
    wr32(at + 36, d.div);
    for (size_t i = 0; i < d.pairs.size(); ++i) {
        wr32(at + 64 + 8 * i, d.pairs[i].addr);
        wr32(at + 68 + 8 * i, d.pairs[i].value);
    }
}

void put_qmd(uint64_t at, const std::vector<Pair>& pairs, uint32_t count) {
    wr32(at, count);
    wr32(at + 4, 0xA5A5A5A5u);
    for (size_t i = 0; i < pairs.size(); ++i) {
        wr32(at + 8 + 8 * i, pairs[i].addr);
        wr32(at + 12 + 8 * i, pairs[i].value);
    }
}

struct Outcome {
    bool launch = false;
    bool skip = false;
    uint16_t skip_n = 0;
    std::vector<Pair> writes;
    bool fault = false;
};

Outcome model_ild(uint64_t at) {
    Outcome o;
    const uint32_t ctl = rd32(at), skip_n = rd32(at + 4) & 0xFFFF;
    uint64_t rec, pred;
    std::memcpy(&rec, &g_mem[at + 8], 8);
    std::memcpy(&pred, &g_mem[at + 16], 8);
    const uint32_t ref = rd32(at + 24), bias = rd32(at + 32), div = rd32(at + 36);
    uint32_t count = rd32(at + 28);
    if (count > 64) {
        count = 64;
    }
    bool exec = true;
    if (ctl & HAS_PRED) {
        exec = (rd32(pred) > ref) != ((ctl & PRED_INV) != 0);
    }
    uint32_t g[3] = {1, 1, 1};
    if (exec && (ctl & HAS_RECORD)) {
        for (int d = 0; d < 3; ++d) {
            const uint32_t sel = (ctl >> (4 * d)) & 0xF;
            uint64_t v = (sel == SEL_ONE) ? 1 : rd32(rec + 4 * sel);
            if (d == 0) {
                v = (v > bias) ? v - bias : 0;
                if (div > 1) {
                    v /= div;
                }
            }
            const uint32_t sh = (ctl >> (12 + 4 * d)) & 0xF;
            g[d] = uint32_t((v + (uint64_t(1) << sh) - 1) >> sh);
            if (g[d] == 0) {
                exec = false;
            }
        }
    }
    if (!exec) {
        o.skip = true;
        o.skip_n = uint16_t(skip_n);
        return o;
    }
    if (ctl & NO_LAUNCH) {
        return o;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t a = rd32(at + 64 + 8 * i) & 0xFFF, v = rd32(at + 68 + 8 * i);
        if ((ctl & HAS_RECORD) && a >= DCR_GRID_X && a <= DCR_GRID_X + 2) {
            v = g[a - DCR_GRID_X];
        }
        o.writes.push_back({a, v});
    }
    o.launch = true;
    return o;
}

Outcome model_qmd(uint64_t at) {
    Outcome o;
    uint32_t count = rd32(at);
    if (count > 64) {
        count = 64;
    }
    for (uint32_t i = 0; i < count; ++i) {
        o.writes.push_back({rd32(at + 8 + 8 * i) & 0xFFF, rd32(at + 12 + 8 * i)});
    }
    o.launch = true;
    return o;
}

using Sim = vl_simulator<VVX_cp_launch_seq_top>;

void set_cmd(Sim& sim, uint8_t opcode, uint64_t arg0) {
    for (int i = 0; i < 9; ++i) {
        sim->cmd_packed[i] = 0;
    }
    sim->cmd_packed[6] = uint32_t(arg0);
    sim->cmd_packed[7] = uint32_t(arg0 >> 32);
    sim->cmd_packed[8] = opcode;
}

struct Rsp { uint64_t addr; uint64_t ready_at; };

// Drive one command to completion and compare with `want`.
void run(Sim& sim, uint64_t& tick, uint8_t opcode, uint64_t arg0, const Outcome& want,
         bool stress) {
    std::deque<Rsp> rq;
    std::vector<Pair> writes;
    uint64_t now = 0, last_write = 0, proxy_busy_until = stress ? rnd(12) : 0;
    int starts = 0, dones = 0, reads = 0;
    bool skipped = false, faulted = false;
    uint16_t skip_n = 0;
    bool busy = false, seen_idle_hold = false;
    uint64_t busy_rise = 0, busy_fall = 0, fall_at = ~0ull;
    bool fired = false;

    for (; now < 20000; ++now) {
        sim->grant = (now == 0);
        set_cmd(sim, opcode, arg0);
        sim->arready = !stress || rnd(4) != 0;
        bool rv = !rq.empty() && rq.front().ready_at <= now;
        sim->rvalid = rv;
        if (rv) {
            const uint64_t a = rq.front().addr;
            for (int w = 0; w < 16; ++w) {
                sim->rdata[w] = rd32(a + 4 * w);
            }
        }
        sim->rid = 0;
        sim->rlast = 1;
        sim->rresp = 0;
        sim->dcr_req_ready = !stress || rnd(3) != 0;
        sim->dcr_proxy_idle = (now >= proxy_busy_until);
        if (busy_rise && now == busy_rise) {
            busy = true;
        }
        if (busy && now == busy_fall) {
            busy = false;
            fall_at = now;
        }
        sim->gpu_busy = busy;
        sim->eval();

        EXPECT(!sim->awvalid && !sim->wvalid, "sequencer issued a write");
        if (sim->arvalid && sim->arready) {
            const uint64_t a = sim->araddr;
            EXPECT((a & 63) == 0, "AR address 0x%llx is not line aligned", (unsigned long long)a);
            EXPECT(sim->arlen == 0 && sim->arsize == 6 && sim->arburst == 1,
                   "AR is not a single 64-byte INCR beat");
            EXPECT(a + 64 <= MEM_SIZE, "AR address 0x%llx out of range", (unsigned long long)a);
            rq.push_back({a, now + 1 + (stress ? rnd(8) : 0)});
            ++reads;
        }
        if (rv && sim->rready) {
            rq.pop_front();
        }
        if (sim->dcr_hold && sim->dcr_proxy_idle) {
            seen_idle_hold = true;
        }
        if (sim->dcr_req_valid && sim->dcr_req_ready) {
            EXPECT(seen_idle_hold, "DCR write before the proxy went idle under hold");
            EXPECT(sim->dcr_hold, "DCR write without the DCR hold");
            writes.push_back({uint32_t(sim->dcr_req_addr), uint32_t(sim->dcr_req_data)});
            last_write = now;
        }
        if (sim->start) {
            ++starts;
            EXPECT(writes.size() == want.writes.size(), "start before the last DCR write");
            EXPECT(writes.empty() || now >= last_write + SETTLE,
                   "start %llu cycles after the last DCR write",
                   (unsigned long long)(now - last_write));
            busy_rise = now + 1 + (stress ? rnd(5) : 0);
            busy_fall = busy_rise + 1 + (stress ? rnd(20) : 3);
        }
        if (sim->done) {
            ++dones;
            skipped = sim->skip;
            skip_n = sim->skip_n;
            faulted = sim->fault;
            EXPECT(!want.launch || (fall_at == now), "done is not on the busy falling edge");
            fired = true;
        }
        tick = sim.step(tick, 2);
        if (fired) {
            break;
        }
    }
    EXPECT(fired, "command never finished");
    EXPECT(rq.empty(), "read response left behind");
    EXPECT(dones == 1, "%d done pulses", dones);
    EXPECT(starts == (want.launch ? 1 : 0), "%d start pulses, expected %d", starts, want.launch ? 1 : 0);
    EXPECT(skipped == want.skip, "skip=%d, expected %d", skipped, want.skip);
    EXPECT(faulted == want.fault, "fault=%d, expected %d", faulted, want.fault);
    if (want.fault) {
        EXPECT(reads == 0, "%d reads of a refused descriptor", reads);
    }
    if (want.skip) {
        EXPECT(skip_n == want.skip_n, "skip_n=%u, expected %u", skip_n, want.skip_n);
    }
    EXPECT(writes.size() == want.writes.size(), "%zu DCR writes, expected %zu",
           writes.size(), want.writes.size());
    for (size_t i = 0; i < writes.size(); ++i) {
        EXPECT(writes[i].addr == want.writes[i].addr && writes[i].value == want.writes[i].value,
               "DCR write %zu = (0x%x, 0x%x), expected (0x%x, 0x%x)", i,
               writes[i].addr, writes[i].value, want.writes[i].addr, want.writes[i].value);
    }
    // A few idle cycles between commands; nothing may happen.
    for (int i = 0; i < 4; ++i) {
        sim->grant = 0;
        sim->rvalid = 0;
        sim->gpu_busy = 0;
        sim->eval();
        EXPECT(!sim->arvalid && !sim->dcr_req_valid && !sim->start && !sim->done,
               "activity while idle");
        tick = sim.step(tick, 2);
    }
}

void run_ild(Sim& sim, uint64_t& tick, uint64_t at, const Ild& d) {
    put_ild(at, d);
    const Outcome want = model_ild(at);
    run(sim, tick, OP_LAUNCH_IND, at, want, false);
    run(sim, tick, OP_LAUNCH_IND, at, want, true);
}

std::vector<Pair> kmu_pairs(uint32_t n) {
    std::vector<Pair> p;
    for (uint32_t i = 0; i < n; ++i) {
        p.push_back({0x010 + (i % 20), 0x1000u * (i + 1) + i});
    }
    return p;
}

// The standard QMD body: entry, args, block, grid, and the rest.
std::vector<Pair> launch_pairs() {
    std::vector<Pair> p = kmu_pairs(18);
    p[9]  = {DCR_GRID_X + 0, 111};
    p[10] = {DCR_GRID_X + 1, 222};
    p[11] = {DCR_GRID_X + 2, 333};
    return p;
}

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Sim sim;
    uint64_t tick = 0;
    sim->grant = 0;
    sim->rvalid = 0;
    sim->dcr_req_ready = 1;
    sim->dcr_proxy_idle = 1;
    sim->gpu_busy = 0;
    tick = sim.reset(tick);
    int cases = 0;

    g_case = "plain launch";
    run(sim, tick, OP_LAUNCH, 0, Outcome{true, false, 0, {}}, false);
    run(sim, tick, OP_LAUNCH, 0, Outcome{true, false, 0, {}}, true);
    ++cases;

    g_case = "qmd";
    for (uint32_t n : {0u, 1u, 7u, 8u, 18u, 64u}) {
        put_qmd(0x2000, kmu_pairs(n), n);
        run(sim, tick, OP_LAUNCH_QMD, 0x2000, model_qmd(0x2000), false);
        run(sim, tick, OP_LAUNCH_QMD, 0x2000, model_qmd(0x2000), true);
        ++cases;
    }
    g_case = "qmd corrupt count";
    put_qmd(0x2000, kmu_pairs(64), 1000);
    run(sim, tick, OP_LAUNCH_QMD, 0x2000, model_qmd(0x2000), true);
    ++cases;

    g_case = "indirect unconditional";
    {
        Ild d;
        d.pairs = launch_pairs();
        run_ild(sim, tick, 0x3000, d);
        d.pairs = kmu_pairs(0);
        run_ild(sim, tick, 0x3000, d);
        d.pairs = kmu_pairs(64);
        run_ild(sim, tick, 0x3000, d);
        d.count_override = 500;
        run_ild(sim, tick, 0x3000, d);
        cases += 4;
    }

    g_case = "predicate";
    for (uint32_t off = 0; off < 64; off += 4) {
        for (int k = 0; k < 4; ++k) {
            Ild d;
            d.ctl = HAS_PRED | ((k & 2) ? PRED_INV : 0);
            d.pred = 0x8000 + off;
            wr32(d.pred, 10);
            d.ref = (k & 1) ? 10 : 9;
            d.skip_n = uint16_t(off + k + 1);
            d.pairs = launch_pairs();
            run_ild(sim, tick, 0x3000, d);
            ++cases;
        }
    }
    g_case = "predicate unsigned";
    {
        Ild d;
        d.ctl = HAS_PRED;
        d.pred = 0x8004;
        wr32(d.pred, 0x80000000u);
        d.ref = 5;
        d.pairs = launch_pairs();
        run_ild(sim, tick, 0x3000, d);
        ++cases;
    }

    g_case = "record grid";
    {
        const uint64_t rec = 0x9000;
        for (int i = 0; i < 16; ++i) {
            wr32(rec + 4 * i, 3 + 2 * i);
        }
        Ild d;
        d.ctl = HAS_RECORD | ctl_sel(0, 1, 2);
        d.rec = rec;
        d.pairs = launch_pairs();
        run_ild(sim, tick, 0x3000, d);
        d.ctl = HAS_RECORD | ctl_sel(5, SEL_ONE, 9) | ctl_sh(3, 0, 1);
        run_ild(sim, tick, 0x3000, d);
        d.ctl = HAS_RECORD | ctl_sel(SEL_ONE, SEL_ONE, SEL_ONE);
        run_ild(sim, tick, 0x3000, d);
        // Grid pairs absent: nothing to substitute.
        d.ctl = HAS_RECORD | ctl_sel(0, 1, 2);
        d.pairs = kmu_pairs(9);
        run_ild(sim, tick, 0x3000, d);
        cases += 4;
    }

    g_case = "zero dimension";
    for (int dim = 0; dim < 3; ++dim) {
        const uint64_t rec = 0x9100;
        wr32(rec + 0, 4);
        wr32(rec + 4, 4);
        wr32(rec + 8, 4);
        wr32(rec + 4 * dim, 0);
        Ild d;
        d.ctl = HAS_RECORD | ctl_sel(0, 1, 2);
        d.rec = rec;
        d.skip_n = uint16_t(7 + dim);
        d.pairs = launch_pairs();
        run_ild(sim, tick, 0x3000, d);
        ++cases;
    }

    g_case = "line-crossing record";
    for (uint32_t off = 0; off < 64; off += 4) {
        const uint64_t rec = 0xA000 + off;
        for (int i = 0; i < 15; ++i) {
            wr32(rec + 4 * i, 1 + i);
        }
        Ild d;
        d.rec = rec;
        d.pairs = launch_pairs();
        d.ctl = HAS_RECORD | ctl_sel(0, 1, 14);
        run_ild(sim, tick, 0x3000, d);
        d.ctl = HAS_RECORD | ctl_sel(14, 7, 2);
        run_ild(sim, tick, 0x3000, d);
        cases += 2;
    }
    g_case = "record at a 4 KiB page end";
    {
        const uint64_t rec = 0xBFF8;
        wr32(rec, 5);
        wr32(rec + 4, 6);
        wr32(rec + 8, 7);
        Ild d;
        d.rec = rec;
        d.ctl = HAS_RECORD | ctl_sel(0, 1, 2);
        d.pairs = launch_pairs();
        run_ild(sim, tick, 0x3000, d);
        ++cases;
    }

    g_case = "x bias and divider";
    {
        struct C { uint32_t v, bias, div, sh; };
        const C cs[] = {
            {1000, 16, 7, 2}, {1000, 16, 7, 0}, {1000, 0, 0, 0}, {1000, 0, 1, 0},
            {1000, 0, 2, 0},  {7, 0, 7, 0},     {6, 0, 7, 0},    {0, 0, 3, 0},
            {0xFFFFFFFFu, 0, 0xFFFFFFFFu, 0},   {0xFFFFFFFEu, 0, 0xFFFFFFFFu, 0},
            {0xFFFFFFFFu, 0, 2, 15},            {0xFFFFFFFFu, 0, 0, 15},
            {5, 5, 0, 0},     {5, 6, 0, 0},     {5, 4, 0, 0},    {1, 0, 0, 15},
        };
        for (const C& c : cs) {
            const uint64_t rec = 0x9200;
            wr32(rec, c.v);
            Ild d;
            d.ctl = HAS_RECORD | ctl_sel(0, SEL_ONE, SEL_ONE) | ctl_sh(c.sh, 0, 0);
            d.rec = rec;
            d.bias = c.bias;
            d.div = c.div;
            d.skip_n = 3;
            d.pairs = launch_pairs();
            run_ild(sim, tick, 0x3000, d);
            ++cases;
        }
    }

    g_case = "predicate-only";
    for (int k = 0; k < 2; ++k) {
        Ild d;
        d.ctl = HAS_PRED | NO_LAUNCH;
        d.pred = 0x8010;
        wr32(d.pred, uint32_t(k));
        d.skip_n = 31;
        run_ild(sim, tick, 0x3000, d);
        ++cases;
    }

    g_case = "misaligned descriptor";
    {
        Outcome refused;
        refused.fault = true;
        put_qmd(0x2000, kmu_pairs(4), 4);
        Ild d;
        d.pairs = launch_pairs();
        put_ild(0x3000, d);
        for (uint64_t off : {4ull, 8ull, 32ull, 60ull}) {
            run(sim, tick, OP_LAUNCH_QMD, 0x2000 + off, refused, false);
            run(sim, tick, OP_LAUNCH_IND, 0x3000 + off, refused, true);
            cases += 2;
        }
        // A plain launch carries no descriptor; its arg0 is ignored.
        run(sim, tick, OP_LAUNCH, 0x1234, Outcome{true, false, 0, {}}, false);
        ++cases;
    }

    g_case = "random";
    for (int it = 0; it < 400; ++it) {
        const uint64_t rec = 0xC000 + 4 * rnd(64);
        const uint64_t pred = 0xD000 + 4 * rnd(64);
        for (int i = 0; i < 15; ++i) {
            const uint32_t r = rnd(4);
            wr32(rec + 4 * i, r == 0 ? 0 : (r == 1 ? rnd(8) : rnd(100000)));
        }
        wr32(pred, rnd(8));
        Ild d;
        d.ctl = (rnd(2) ? HAS_RECORD : 0) | (rnd(2) ? HAS_PRED : 0)
              | (rnd(2) ? PRED_INV : 0) | (rnd(5) == 0 ? NO_LAUNCH : 0)
              | ctl_sel(rnd(16), rnd(16), rnd(16)) | ctl_sh(rnd(16), rnd(16), rnd(16));
        d.rec = rec;
        d.pred = pred;
        d.ref = rnd(8);
        d.bias = rnd(3) ? 0 : rnd(1000);
        d.div = rnd(3) ? 0 : rnd(50);
        d.skip_n = uint16_t(rnd(65536));
        d.pairs = rnd(3) ? launch_pairs() : kmu_pairs(rnd(65));
        const uint64_t at = 0x4000 + 64 * rnd(64);
        put_ild(at, d);
        run(sim, tick, OP_LAUNCH_IND, at, model_ild(at), true);
        ++cases;
    }

    std::printf("PASSED — %d launch sequencer cases\n", cases);
    return 0;
}
