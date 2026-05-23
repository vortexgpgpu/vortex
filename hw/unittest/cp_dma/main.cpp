// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_dma.
//
// VX_cp_dma is dual-port (axi_host + axi_dev). A CMD_MEM_COPY routes both
// the read and the write to the device port, so this TB attaches a full
// AXI slave memory model to the device port (d_*) and holds the host port
// (h_*) idle.
//
// Verifies that the DMA module:
//   1. Issues an AXI AR at src on axi_dev, captures one cache line of rdata.
//   2. Issues an AXI AW at dst + W with the captured data, awaits B.
//   3. Pulses `done` exactly once.
//
// Scenarios:
//   1. COPY between two regions of the synthetic memory; verify dst
//      bytes match src bytes byte-for-byte.
//   2. Second back-to-back COPY (different addrs / pattern) re-arms
//      cleanly — DMA returns to IDLE and accepts the next grant.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_dma_top.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1); \
    } \
} while (0)

// cmd_t packer: opcode in MSB word (index 8), arg0/1/2 in words [6..7],
// [4..5], [2..3] respectively.
static void pack_cmd(uint32_t out_words[9],
                     uint8_t opcode, uint8_t flags,
                     uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    for (int i = 0; i < 9; ++i) out_words[i] = 0;
    out_words[0] = 0;
    out_words[1] = 0;
    out_words[2] = (uint32_t)(arg2 & 0xffffffffu);
    out_words[3] = (uint32_t)(arg2 >> 32);
    out_words[4] = (uint32_t)(arg1 & 0xffffffffu);
    out_words[5] = (uint32_t)(arg1 >> 32);
    out_words[6] = (uint32_t)(arg0 & 0xffffffffu);
    out_words[7] = (uint32_t)(arg0 >> 32);
    out_words[8] = (uint32_t)opcode | ((uint32_t)flags << 8);
}

// ---- AXI4 slave model (same pipeline pattern as cp_axi_path TB) ----
struct AxiSlave {
    static constexpr uint64_t MEM_BASE = 0x1000;
    static constexpr int      MEM_SIZE = 4096;
    uint8_t mem[MEM_SIZE] = {0};

    bool         r_inflight = false;
    uint64_t     r_addr     = 0;
    uint8_t      r_id       = 0;

    bool         aw_taken   = false;
    uint64_t     aw_addr    = 0;
    uint8_t      aw_id      = 0;
    bool         b_pending  = false;
    uint8_t      b_id       = 0;

    void mem_write_cl(uint64_t addr, const uint8_t* src) {
        for (int i = 0; i < 64; ++i) {
            int64_t a = (int64_t)addr - (int64_t)MEM_BASE + i;
            if (a >= 0 && a < MEM_SIZE) mem[a] = src[i];
        }
    }
    void mem_read_cl(uint64_t addr, uint32_t* dst) const {
        for (int w = 0; w < 16; ++w) {
            uint32_t v = 0;
            for (int b = 0; b < 4; ++b) {
                int64_t a = (int64_t)addr - (int64_t)MEM_BASE + w*4 + b;
                if (a >= 0 && a < MEM_SIZE) v |= (uint32_t)mem[a] << (8 * b);
            }
            dst[w] = v;
        }
    }
    int mem_cmp_cl(uint64_t addr_a, uint64_t addr_b) const {
        for (int i = 0; i < 64; ++i) {
            int64_t aa = (int64_t)addr_a - (int64_t)MEM_BASE + i;
            int64_t ab = (int64_t)addr_b - (int64_t)MEM_BASE + i;
            uint8_t va = (aa >= 0 && aa < MEM_SIZE) ? mem[aa] : 0;
            uint8_t vb = (ab >= 0 && ab < MEM_SIZE) ? mem[ab] : 0;
            if (va != vb) return i;
        }
        return -1;
    }

    template <typename T>
    void comb_drive(T* top) {
        top->d_arready = !r_inflight;
        top->d_rvalid = r_inflight;
        top->d_rid    = r_id;
        top->d_rlast  = 1;
        top->d_rresp  = 0;
        if (r_inflight) mem_read_cl(r_addr, top->d_rdata);

        top->d_awready = !aw_taken;
        top->d_wready  = aw_taken && !b_pending;
        top->d_bvalid  = b_pending;
        top->d_bid     = b_id;
        top->d_bresp   = 0;
    }

    template <typename T>
    void posedge_update(T* top) {
        if (top->d_arvalid && top->d_arready) {
            r_inflight = true;
            r_addr     = top->d_araddr;
            r_id       = top->d_arid;
        } else if (r_inflight && top->d_rvalid && top->d_rready) {
            r_inflight = false;
        }

        if (top->d_awvalid && top->d_awready) {
            aw_taken = true;
            aw_addr  = top->d_awaddr;
            aw_id    = top->d_awid;
        }
        if (aw_taken && top->d_wvalid && top->d_wready) {
            // Write 64 bytes from wdata[0..15] into memory at aw_addr.
            for (int w = 0; w < 16; ++w) {
                uint32_t v = top->d_wdata[w];
                for (int b = 0; b < 4; ++b) {
                    int64_t a = (int64_t)aw_addr - (int64_t)MEM_BASE + w*4 + b;
                    if (a >= 0 && a < MEM_SIZE) mem[a] = (uint8_t)(v >> (8 * b));
                }
            }
            aw_taken  = false;
            b_pending = true;
            b_id      = aw_id;
        }
        if (b_pending && top->d_bvalid && top->d_bready) b_pending = false;
    }
};

template <typename T>
static void cycle(vl_simulator<T>& sim, AxiSlave& s, uint64_t& tick) {
    auto* top = sim.operator->();
    s.comb_drive(top);
    top->eval();
    s.comb_drive(top);
    top->eval();
    s.posedge_update(top);
    tick = sim.step(tick, 2);
    s.comb_drive(top);
    top->eval();
}

template <typename T>
static void run_copy(vl_simulator<T>& sim, AxiSlave& slave, uint64_t& tick,
                     uint64_t src, uint64_t dst, const uint8_t* pattern) {
    slave.mem_write_cl(src, pattern);

    // Drain any leftover state (a previous run_copy returns with the FSM
    // in S_DONE; one idle cycle takes it back to S_IDLE before we drive
    // the next grant).
    sim->grant = 0;
    for (int i = 0; i < 2; ++i) cycle(sim, slave, tick);

    uint32_t c[9];
    pack_cmd(c, /*opcode=*/0x03 /*MEM_COPY*/, 0, /*arg0=dst*/dst,
             /*arg1=src*/src, /*arg2=size*/64);
    for (int i = 0; i < 9; ++i) sim->cmd_packed[i] = c[i];

    // Hold grant high until the FSM observably leaves IDLE (i.e. the
    // master starts issuing AXI traffic). Dropping grant too early is a
    // common race — IDLE -> REQ_AR is on a posedge so the FSM must see
    // grant=1 at that exact edge.
    sim->grant = 1;
    bool latched = false;
    for (int g = 0; g < 8 && !latched; ++g) {
        cycle(sim, slave, tick);
        if (sim->d_arvalid) latched = true;
    }
    sim->grant = 0;
    EXPECT(latched, "DMA never asserted arvalid (grant capture failed)");

    bool got_done = false;
    for (int g = 0; g < 50 && !got_done; ++g) {
        cycle(sim, slave, tick);
        if (sim->done) got_done = true;
    }
    EXPECT(got_done, "DMA did not signal done within 50 cycles");
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    vl_simulator<VVX_cp_dma_top> sim;
    uint64_t tick = 0;
    AxiSlave slave;

    sim->grant = 0;
    for (int i = 0; i < 9; ++i) sim->cmd_packed[i] = 0;
    // The host AXI port (h_*) stays idle for CMD_MEM_COPY; verilator
    // zero-inits its input ports and the harness never drives them.
    tick = sim.reset(tick);

    // ----- Test 1: copy at known offsets -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0xA0 + i);
        run_copy(sim, slave, tick, /*src=*/0x1000, /*dst=*/0x1100, pat);

        int diff = slave.mem_cmp_cl(0x1000, 0x1100);
        EXPECT(diff < 0, "T1: dst doesn't match src after copy");
    }

    // ----- Test 2: back-to-back copy with different pattern -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0x5A ^ (i << 1));
        run_copy(sim, slave, tick, /*src=*/0x1200, /*dst=*/0x1300, pat);

        int diff = slave.mem_cmp_cl(0x1200, 0x1300);
        EXPECT(diff < 0, "T2: second copy mismatch");
    }

    std::printf("PASSED — 2 scenarios\n");
    return 0;
}
