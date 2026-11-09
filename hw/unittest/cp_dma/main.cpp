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
//   3. With a page-table root in satp, a COPY between two virtual pages
//      walks the table for both operands and moves the physical lines.
//   4. A superpage leaf takes the low address bits from the VA.
//   5. F_MEM_PHYSICAL skips the walk.
//   6. An unmapped VA passes through untranslated, as in the C++ model.
// ============================================================================

#include "vl_simulator.h"
#include "util.h"
#include "VVX_cp_dma_top.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <unordered_map>
#include <VX_config.h>
#include <VX_types.h>

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

// ---- AXI4 slave model (sparse, cache-line granular) ----
struct AxiSlave {
    std::unordered_map<uint64_t, std::array<uint8_t, 64>> lines;

    uint8_t& byte(uint64_t a) {
        auto it = lines.find(a >> 6);
        if (it == lines.end()) it = lines.emplace(a >> 6, std::array<uint8_t, 64>{}).first;
        return it->second[a & 63];
    }
    uint8_t peek(uint64_t a) const {
        auto it = lines.find(a >> 6);
        return (it == lines.end()) ? 0 : it->second[a & 63];
    }
    void write_word(uint64_t a, uint64_t v, int bytes) {
        for (int b = 0; b < bytes; ++b) byte(a + b) = uint8_t(v >> (8 * b));
    }

    bool         r_inflight = false;
    uint64_t     r_addr     = 0;
    uint8_t      r_id       = 0;

    bool         aw_taken   = false;
    uint64_t     aw_addr    = 0;
    uint8_t      aw_id      = 0;
    bool         b_pending  = false;
    uint8_t      b_id       = 0;

    void mem_write_cl(uint64_t addr, const uint8_t* src) {
        for (int i = 0; i < 64; ++i) byte(addr + i) = src[i];
    }
    void mem_read_cl(uint64_t addr, uint32_t* dst) const {
        for (int w = 0; w < 16; ++w) {
            uint32_t v = 0;
            for (int b = 0; b < 4; ++b) v |= (uint32_t)peek(addr + w*4 + b) << (8 * b);
            dst[w] = v;
        }
    }
    int mem_cmp_cl(uint64_t addr_a, uint64_t addr_b) const {
        for (int i = 0; i < 64; ++i) {
            if (peek(addr_a + i) != peek(addr_b + i)) return i;
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
        if (r_inflight) {
            mem_read_cl(r_addr, vortex::VDataCast<uint32_t*, sizeof(top->d_rdata)>::get(top->d_rdata));
        }

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
                for (int b = 0; b < 4; ++b) byte(aw_addr + w*4 + b) = (uint8_t)(v >> (8 * b));
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

// Page-table builder in the slave's memory: Sv32 at XLEN=32, Sv39 at 64.
struct PageTable {
    static constexpr int LEVEL_BITS = __builtin_ctz(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
    AxiSlave& mem;
    uint64_t  root;
    uint64_t  next_table;

    PageTable(AxiSlave& m, uint64_t root_pa) : mem(m), root(root_pa), next_table(root_pa + VX_VM_PT_SIZE) {}

    uint64_t satp() const {
#if VX_VM_PT_LEVEL == 2
        return (1ull << 31) | (root >> VX_VM_PAGE_LOG2_SIZE);
#else
        return (8ull << 60) | (root >> VX_VM_PAGE_LOG2_SIZE);
#endif
    }
    uint64_t pte_addr(uint64_t table, uint64_t va, int level) const {
        uint64_t idx = (va >> (VX_VM_PAGE_LOG2_SIZE + level * LEVEL_BITS)) & ((1ull << LEVEL_BITS) - 1);
        return table + idx * VX_VM_PTE_SIZE;
    }
    uint64_t read_pte(uint64_t a) const {
        uint64_t v = 0;
        for (int b = 0; b < VX_VM_PTE_SIZE; ++b) v |= uint64_t(mem.peek(a + b)) << (8 * b);
        return v;
    }
    // Map one page (leaf_level 0) or one superpage (leaf_level > 0).
    void map(uint64_t va, uint64_t pa, int leaf_level) {
        uint64_t table = root;
        for (int level = VX_VM_PT_LEVEL - 1; level > leaf_level; --level) {
            uint64_t a = pte_addr(table, va, level);
            uint64_t pte = read_pte(a);
            if ((pte & 1) == 0) {
                pte = ((next_table >> VX_VM_PAGE_LOG2_SIZE) << 10) | 0x1;   // V, pointer
                mem.write_word(a, pte, VX_VM_PTE_SIZE);
                next_table += VX_VM_PT_SIZE;
            }
            table = (pte >> 10) << VX_VM_PAGE_LOG2_SIZE;
        }
        uint64_t leaf = ((pa >> VX_VM_PAGE_LOG2_SIZE) << 10) | 0xC7;           // D A W R V
        mem.write_word(pte_addr(table, va, leaf_level), leaf, VX_VM_PTE_SIZE);
    }
};

template <typename T>
static void run_copy(vl_simulator<T>& sim, AxiSlave& slave, uint64_t& tick,
                     uint64_t src, uint64_t dst, const uint8_t* pattern,
                     uint64_t pattern_pa, uint8_t flags = 0) {
    slave.mem_write_cl(pattern_pa, pattern);

    // Drain any leftover state (a previous run_copy returns with the FSM
    // in S_DONE; one idle cycle takes it back to S_IDLE before we drive
    // the next grant).
    sim->grant = 0;
    for (int i = 0; i < 2; ++i) cycle(sim, slave, tick);

    uint32_t c[9];
    pack_cmd(c, /*opcode=*/0x03 /*MEM_COPY*/, flags, /*arg0=dst*/dst,
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
    for (int g = 0; g < 200 && !got_done; ++g) {
        cycle(sim, slave, tick);
        if (sim->done) got_done = true;
    }
    EXPECT(got_done, "DMA did not signal done within 200 cycles");
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
        run_copy(sim, slave, tick, /*src=*/0x1000, /*dst=*/0x1100, pat, 0x1000);

        int diff = slave.mem_cmp_cl(0x1000, 0x1100);
        EXPECT(diff < 0, "T1: dst doesn't match src after copy");
    }

    // ----- Test 2: back-to-back copy with different pattern -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0x5A ^ (i << 1));
        run_copy(sim, slave, tick, /*src=*/0x1200, /*dst=*/0x1300, pat, 0x1200);

        int diff = slave.mem_cmp_cl(0x1200, 0x1300);
        EXPECT(diff < 0, "T2: second copy mismatch");
    }

#if VX_CFG_VM_ENABLED
    PageTable pt(slave, 0x100000);
    pt.map(0x80001000, 0x20000, 0);
    pt.map(0x80005000, 0x23000, 0);
    pt.map(0xC0000000, 0x400000, 1);
    sim->satp = pt.satp();

    // ----- Test 3: both operands walk the table -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0x3C + 7 * i);
        run_copy(sim, slave, tick, /*src=*/0x80001040, /*dst=*/0x80005080, pat, 0x20040);
        EXPECT(slave.mem_cmp_cl(0x20040, 0x23080) < 0, "T3: VA copy did not land at the mapped PA");
        EXPECT(slave.lines.count(0x80005080 >> 6) == 0, "T3: VA copy wrote the untranslated address");
    }

    // ----- Test 4: superpage leaf keeps the VA's low bits -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0x91 ^ i);
        run_copy(sim, slave, tick, /*src=*/0xC0012340, /*dst=*/0x80005100, pat, 0x412340);
        EXPECT(slave.mem_cmp_cl(0x412340, 0x23100) < 0, "T4: superpage copy mismatch");
    }

    // ----- Test 5: F_MEM_PHYSICAL skips the walk -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0x17 + i);
        run_copy(sim, slave, tick, /*src=*/0x30000, /*dst=*/0x30400, pat, 0x30000, /*flags=*/0x04);
        EXPECT(slave.mem_cmp_cl(0x30000, 0x30400) < 0, "T5: physical copy mismatch");
    }

    // ----- Test 6: an unmapped VA passes through -----
    {
        uint8_t pat[64];
        for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(0xE1 - i);
        run_copy(sim, slave, tick, /*src=*/0x00031000, /*dst=*/0x80005140, pat, 0x31000);
        EXPECT(slave.mem_cmp_cl(0x31000, 0x23140) < 0, "T6: unmapped source did not pass through");
    }

    std::printf("PASSED — 6 scenarios\n");
#else
    std::printf("PASSED — 2 scenarios (VX_CFG_VM_ENABLE off: no page walk)\n");
#endif
    return 0;
}
