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

`include "VX_define.vh"

// Per-core MMU: a D-side and an I-side L1 TLB stage, a shared page-table
// walker fed by both, and a merge that folds the walker's PTE-fetch port
// into the dcache path. The itlb output goes straight to the icache. `empty`
// aggregates all stage/miss-station/walker state for kernel-drain and
// barrier ordering. The walker's structural faults and the L1s' permission
// faults drive a fault sideband for the fault-latch surface.
module VX_mmu import VX_gpu_pkg::*, VX_tlb_pkg::*; (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    input wire [`VX_CFG_XLEN-1:0] satp,

    VX_mem_bus_if.slave  lsu_dcache_if  [DCACHE_NUM_REQS],
    VX_mem_bus_if.master dcache_mem_if  [DCACHE_NUM_REQS],

    VX_mem_bus_if.slave  lsu_icache_if  [1],
    VX_mem_bus_if.master icache_mem_if  [1],

    output wire          empty
);
    localparam NR       = DCACHE_NUM_REQS;
    localparam L1_ID_W  = `CLOG2(`VX_CFG_L1_TLB_MSHR_SIZE);
    localparam PTW_ID_W = L1_ID_W + `ARB_SEL_BITS(2, 1);
    localparam TAGW_BASE = DCACHE_TAG_WIDTH_BASE;

    // ---------------------------------------------------------------------
    // TLB miss/fill fabric
    // ---------------------------------------------------------------------
    VX_tlb_bus_if #(.ID_WIDTH (L1_ID_W))  l1_tbus [2] ();
    VX_tlb_bus_if #(.ID_WIDTH (PTW_ID_W)) ptw_tbus ();

    VX_tlb_flush_if flush_l1d ();
    VX_tlb_flush_if flush_l1i ();
    VX_tlb_flush_if flush_ptw ();
    assign flush_l1d.req = 1'b0;
    assign flush_l1i.req = 1'b0;
    assign flush_ptw.req = 1'b0;
    `UNUSED_VAR (flush_l1d.done)
    `UNUSED_VAR (flush_l1i.done)
    `UNUSED_VAR (flush_ptw.done)

    // ---------------------------------------------------------------------
    // D-side L1 TLB (translated lanes merged with the walker below)
    // ---------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (TAGW_BASE)
    ) dtlb_out [NR] ();

    wire dtlb_empty, itlb_empty, ptw_empty;
    wire dtlb_fault_valid, itlb_fault_valid;

`ifdef PERF_ENABLE
    mmu_perf_t dtlb_perf, itlb_perf;
`endif

    VX_tlb_l1 #(
        .INSTANCE_ID ("dtlb"),
        .NUM_LANES   (NR),
        .TLB_SIZE    (`VX_CFG_DTLB_SIZE),
        .EXEC_SIDE   (0),
        .DATA_SIZE   (DCACHE_WORD_SIZE),
        .TAG_WIDTH   (TAGW_BASE)
    ) dtlb (
        .clk         (clk),
        .reset       (reset),
        .satp        (satp),
    `ifdef PERF_ENABLE
        .mmu_perf    (dtlb_perf),
    `endif
        .core_bus_if (lsu_dcache_if),
        .mem_bus_if  (dtlb_out),
        .tlb_bus_if  (l1_tbus[0]),
        .flush_if    (flush_l1d),
        .fault_valid (dtlb_fault_valid),
        `UNUSED_PIN (fault_vpn),
        `UNUSED_PIN (fault_access),
        .empty       (dtlb_empty)
    );

    VX_tlb_l1 #(
        .INSTANCE_ID ("itlb"),
        .NUM_LANES   (1),
        .TLB_SIZE    (`VX_CFG_ITLB_SIZE),
        .EXEC_SIDE   (1),
        .DATA_SIZE   (ICACHE_WORD_SIZE),
        .TAG_WIDTH   (ICACHE_TAG_WIDTH)
    ) itlb (
        .clk         (clk),
        .reset       (reset),
        .satp        (satp),
    `ifdef PERF_ENABLE
        .mmu_perf    (itlb_perf),
    `endif
        .core_bus_if (lsu_icache_if),
        .mem_bus_if  (icache_mem_if),
        .tlb_bus_if  (l1_tbus[1]),
        .flush_if    (flush_l1i),
        .fault_valid (itlb_fault_valid),
        `UNUSED_PIN (fault_vpn),
        `UNUSED_PIN (fault_access),
        .empty       (itlb_empty)
    );

    `UNUSED_VAR (dtlb_fault_valid)
    `UNUSED_VAR (itlb_fault_valid)

    // ---------------------------------------------------------------------
    // Shared walker fed by both L1s
    // ---------------------------------------------------------------------
    VX_tlb_arb #(
        .NUM_INPUTS   (2),
        .ID_WIDTH_IN  (L1_ID_W),
        .OUT_BUF      (1)
    ) tlb_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (l1_tbus),
        .bus_out_if (ptw_tbus)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (TAGW_BASE)
    ) ptw_mem ();

    VX_ptw #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (TAGW_BASE),
        .ID_WIDTH  (PTW_ID_W)
    ) ptw (
        .clk        (clk),
        .reset      (reset),
        .satp       (satp),
        .miss_if    (ptw_tbus),
        .mem_bus_if (ptw_mem),
        .flush_if   (flush_ptw),
        .empty      (ptw_empty)
    );

    // ---------------------------------------------------------------------
    // Fold the shared walker's PTE-fetch port into dcache lane 0 through a
    // 2:1 arbiter; the remaining lanes pass through with the tag widened to
    // the dcache width. (A single (NR+1)->NR arbiter is avoided: its
    // asymmetric response demux does not round-trip the select bit.)
    // ---------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (TAGW_BASE)
    ) arb0_in [2] ();

    `ASSIGN_VX_MEM_BUS_IF (arb0_in[0], dtlb_out[0]);
    `ASSIGN_VX_MEM_BUS_IF (arb0_in[1], ptw_mem);

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) merged0 [1] ();

    VX_mem_bus_arb #(
        .NUM_INPUTS  (2),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (DCACHE_WORD_SIZE),
        .TAG_WIDTH   (TAGW_BASE),
        .TAG_SEL_IDX (TAGW_BASE),
        .ARBITER     ("R"),
        .REQ_OUT_BUF (0),
        .RSP_OUT_BUF (0)
    ) ptw_merge_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (arb0_in),
        .bus_out_if (merged0)
    );

    `ASSIGN_VX_MEM_BUS_IF (dcache_mem_if[0], merged0[0]);
    for (genvar i = 1; i < NR; ++i) begin : g_lane_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (dcache_mem_if[i], dtlb_out[i], DCACHE_TAG_WIDTH, TAGW_BASE, UUID_WIDTH);
    end

    assign empty = dtlb_empty && itlb_empty && ptw_empty;

    // ---------------------------------------------------------------------
    // Performance counters
    // ---------------------------------------------------------------------
`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_ptw_latency;
    always @(posedge clk) begin
        if (reset) begin
            perf_ptw_latency <= '0;
        end else if (~ptw_empty) begin
            perf_ptw_latency <= perf_ptw_latency + PERF_CTR_BITS'(1);
        end
    end

    assign mmu_perf.tlb_reads     = dtlb_perf.tlb_reads     + itlb_perf.tlb_reads;
    assign mmu_perf.tlb_hits      = dtlb_perf.tlb_hits      + itlb_perf.tlb_hits;
    assign mmu_perf.tlb_misses    = dtlb_perf.tlb_misses    + itlb_perf.tlb_misses;
    assign mmu_perf.tlb_evictions = dtlb_perf.tlb_evictions + itlb_perf.tlb_evictions;
    assign mmu_perf.ptw_walks     = dtlb_perf.ptw_walks     + itlb_perf.ptw_walks;
    assign mmu_perf.ptw_latency   = perf_ptw_latency;
`endif

endmodule
