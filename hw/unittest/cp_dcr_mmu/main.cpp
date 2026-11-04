// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test of the CP's DCR read path with the device MMU on it:
// VX_cp_dcr_proxy -> VX_cp_gpu_slice -> VX_mmu_dcr -> cluster DCR bus.
//
// The harness plays the cores (VX_dcr_data): a read of MPM_VALUE for core c
// is answered a few cycles later, a flush read once that core's flush has
// finished, each by a one-cycle response pulse. It drives the command
// sequence the runtime issues after a launch:
//   1. the MMU fault check: CMD_DCR_READ FAULT_INFO (a fault is reported);
//   2. back-to-back perf-counter reads of the cores (CMD_DCR_READ MPM_VALUE);
//   3. FAULT_VA / FAULT_VA_HI reads followed by core reads;
//   4. an MMU read, then CMD_CACHE_FLUSH over the cores: the flush may not
//      retire before the last core has answered, and every core's read is
//      answered by that core.
// Every read must return its own response, and the Vortex DCR port must
// answer every read with exactly one one-cycle pulse (the protocol the
// direct hosts -- rtlsim, FireSim, Foldsim -- sample as well). A response that outlives its
// pulse (the MMU's used to be held until the next DCR access) answers the
// proxy's next read: the slice delays that read's request past the start of
// the proxy's wait.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_dcr_mmu_top.h"
#include <VX_types.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

static int failures = 0;
#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while (0)

enum CmdOp : uint8_t {
    OP_DCR_WRITE   = 0x04,
    OP_DCR_READ    = 0x05,
    OP_CACHE_FLUSH = 0x0A,
};

static constexpr uint32_t NCORES      = 2;
static constexpr uint32_t MPM_LATENCY = 4;   // core CSR read -> response pulse
static constexpr uint64_t FAULT_VA    = 0x00000012345ff000ull;
static constexpr uint32_t FAULT_ACC   = 1;   // write
static constexpr uint32_t FAULT_WORD  = VX_MMU_FAULT_VALID | (FAULT_ACC << VX_MMU_FAULT_ACCESS_SH);

static void set_cmd(VVX_cp_dcr_mmu_top* top, uint8_t opcode, uint64_t arg0, uint64_t arg1, uint8_t flags = 0) {
    uint32_t w[9] = {0};
    w[4] = uint32_t(arg1);  w[5] = uint32_t(arg1 >> 32);
    w[6] = uint32_t(arg0);  w[7] = uint32_t(arg0 >> 32);
    w[8] = uint32_t(opcode) | (uint32_t(flags) << 8);
    for (int i = 0; i < 9; ++i) top->cmd_packed[i] = w[i];
}

// The cores' side of the DCR bus (VX_dcr_data's protocol).
struct Cores {
    struct Rsp { uint64_t at; uint32_t data; uint32_t cid; };
    std::deque<Rsp> sched;           // response pulses to drive, in time order
    uint64_t flush_start = 0;        // cycle the broadcast flush write was seen
    bool     flushing    = false;
    uint32_t mpm_count[NCORES] = {};
    std::vector<uint32_t> answered;  // core of every pulse driven, in order

    static uint32_t mpm_value(uint32_t cid, uint32_t n) { return 0xA0000000u | (cid << 16) | (n + 1); }
    uint64_t flush_done(uint32_t cid) const { return flush_start + 40 * (cid + 1); }

    void request(uint64_t t, bool rw, uint32_t addr, uint32_t data) {
        uint32_t cid = data & 0xffff;
        if (addr == VX_DCR_BASE_CACHE_FLUSH) {
            if (rw) { flush_start = t; flushing = true; return; }
            uint64_t at = flush_done(cid);
            sched.push_back({(at > t) ? at : t + 1, 0, cid});
        } else if (addr == VX_DCR_BASE_MPM_VALUE && !rw) {
            sched.push_back({t + MPM_LATENCY, mpm_value(cid, mpm_count[cid]++), cid});
        }
    }
};

struct Bench {
    vl_simulator<VVX_cp_dcr_mmu_top> sim;
    uint64_t tick = 0, t = 0;
    Cores cores;
    uint32_t mmu_reads_seen = 0;
    uint32_t reads_issued = 0;       // reads reaching the Vortex DCR port
    uint32_t vx_pulses = 0;          // response pulses on the Vortex DCR port
    bool     vx_rsp_prev = false;

    // One clock: drive this cycle's cluster response, sample the cluster
    // request, advance.
    void cycle() {
        bool pulse = !cores.sched.empty() && cores.sched.front().at == t;
        sim->cl_rsp_valid = pulse;
        sim->cl_rsp_data  = pulse ? cores.sched.front().data : 0xdeadbeef;
        if (pulse) { cores.answered.push_back(cores.sched.front().cid); cores.sched.pop_front(); }
        sim->eval();
        bool vx_rsp = sim->vx_rsp_valid;
        EXPECT(!(vx_rsp && vx_rsp_prev), "Vortex DCR response valid held past one cycle at cycle %llu", (unsigned long long)t);
        if (vx_rsp && !vx_rsp_prev) ++vx_pulses;
        vx_rsp_prev = vx_rsp;
        if (sim->cl_req_valid) {
            uint32_t addr = sim->cl_req_addr;
            if (!sim->cl_req_rw) ++reads_issued;
            if (addr >= VX_DCR_MMU_STATE_BEGIN && addr < VX_DCR_MMU_STATE_END && !sim->cl_req_rw) ++mmu_reads_seen;
            cores.request(t, sim->cl_req_rw, addr, sim->cl_req_data);
        }
        tick = sim.step(tick, 2);
        ++t;
    }

    // Issue one CP command and run it to retirement; returns its read value.
    uint32_t run(uint8_t op, uint64_t arg0, uint64_t arg1 = 0, uint8_t flags = 0, uint64_t* done_at = nullptr) {
        set_cmd(sim.operator->(), op, arg0, arg1, flags);
        sim->grant = 1;
        cycle();
        sim->grant = 0;
        for (int i = 0; i < 2000; ++i) {
            sim->eval();
            if (sim->done) {
                uint32_t v = sim->last_rsp_data;
                if (done_at) *done_at = t;
                cycle();
                return v;
            }
            cycle();
        }
        EXPECT(false, "command 0x%x (arg0 0x%llx) never retired", op, (unsigned long long)arg0);
        return 0;
    }
    uint32_t read(uint32_t addr, uint32_t data = 0) { return run(OP_DCR_READ, addr, data); }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Bench b;
    auto& sim = b.sim;
    sim->grant = 0;
    sim->cl_rsp_valid = 0;
    sim->cl_rsp_data = 0;
    sim->fault_valid = 0;
    sim->fault_va = (sizeof(sim->fault_va) == 8) ? FAULT_VA : uint32_t(FAULT_VA);
    sim->fault_access = FAULT_ACC;
    sim->gpu_busy = 0;
    b.tick = sim.reset(b.tick);
    for (int i = 0; i < 4; ++i) b.cycle();

    // The device walker reports a fault during the launch.
    sim->fault_valid = 1;
    b.cycle();
    sim->fault_valid = 0;
    for (int i = 0; i < 4; ++i) b.cycle();

    // 1. the runtime's post-launch fault check
    uint32_t v = b.read(VX_DCR_MMU_FAULT_INFO);
    EXPECT(v == FAULT_WORD, "FAULT_INFO read 0x%x, expected 0x%x", v, FAULT_WORD);

    // 2. perf-counter reads right behind it (vx_device_mpm_query: lo, hi per core)
    for (int round = 0; round < 2; ++round) {
        for (uint32_t cid = 0; cid < NCORES; ++cid) {
            uint32_t exp = Cores::mpm_value(cid, b.cores.mpm_count[cid]);
            v = b.read(VX_DCR_BASE_MPM_VALUE, cid);
            EXPECT(v == exp, "MPM read core %u read 0x%x, expected 0x%x", cid, v, exp);
        }
    }

    // 3. the fault VA, then core reads again
    v = b.read(VX_DCR_MMU_FAULT_VA);
    EXPECT(v == uint32_t(FAULT_VA), "FAULT_VA read 0x%x, expected 0x%x", v, uint32_t(FAULT_VA));
    v = b.read(VX_DCR_MMU_FAULT_VA_HI);
    // XLEN 64 carries the VA high word; XLEN 32 reads it as 0.
    uint32_t exp_hi = (sizeof(sim->fault_va) == 8) ? uint32_t(FAULT_VA >> 32) : 0;
    EXPECT(v == exp_hi, "FAULT_VA_HI read 0x%x, expected 0x%x", v, exp_hi);
    for (uint32_t cid = 0; cid < NCORES; ++cid) {
        uint32_t exp = Cores::mpm_value(cid, b.cores.mpm_count[cid]);
        v = b.read(VX_DCR_BASE_MPM_VALUE, cid);
        EXPECT(v == exp, "MPM read after FAULT_VA, core %u read 0x%x, expected 0x%x", cid, v, exp);
    }

    // 4. an MMU read, then a cache flush over the cores
    v = b.read(VX_DCR_MMU_FAULT_INFO);
    EXPECT(v == FAULT_WORD, "second FAULT_INFO read 0x%x, expected 0x%x", v, FAULT_WORD);
    size_t before = b.cores.answered.size();
    uint64_t done_at = 0;
    b.run(OP_CACHE_FLUSH, NCORES, 0, 0, &done_at);
    uint64_t last_done = b.cores.flush_done(NCORES - 1);
    EXPECT(done_at > last_done, "flush retired at cycle %llu, before the last core finished at %llu",
           (unsigned long long)done_at, (unsigned long long)last_done);
    EXPECT(b.cores.answered.size() - before == NCORES, "flush took %zu core answers, expected %u",
           b.cores.answered.size() - before, NCORES);
    for (int i = 0; i < 100; ++i) b.cycle();
    EXPECT(b.cores.sched.empty(), "a core answer was still pending after the flush retired");

    // One response per read on the Vortex DCR port.
    EXPECT(b.vx_pulses == b.reads_issued, "%u response pulses for %u reads", b.vx_pulses, b.reads_issued);

    // Every MMU read reached the MMU exactly once.
    EXPECT(b.mmu_reads_seen == 4, "the cluster saw %u MMU reads, expected 4", b.mmu_reads_seen);

    if (failures) {
        std::printf("FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
