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

`ifdef VX_CFG_EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_core import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
`endif

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    icache_bus_if,

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_om_bus_if.master     om_bus_if,
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if.slave  raster_bus_if,
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_dcr_flush_if.master  cluster_flush_if,
`endif

    // KMU bus
    VX_kmu_bus_if.slave     kmu_bus_if,

    // Global barrier
    VX_gbar_bus_if.master   gbar_bus_if,

    // Status
    output wire             busy
);
    VX_schedule_if      schedule_if();
    VX_fetch_if         fetch_if();
    VX_decode_if        decode_if();
    VX_sched_csr_if     sched_csr_if();
    VX_decode_sched_if  decode_sched_if();
    VX_issue_sched_if   issue_sched_if[`VX_CFG_ISSUE_WIDTH]();
    VX_commit_sched_if  commit_sched_if();
    VX_branch_ctl_if    branch_ctl_if[`VX_CFG_NUM_ALU_BLOCKS]();
    VX_warp_ctl_if      warp_ctl_if();

    VX_dispatch_if      dispatch_if[NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH]();
    VX_commit_if        commit_if[NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH]();
    VX_writeback_if     writeback_if[`VX_CFG_ISSUE_WIDTH]();

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_txbar_bus_if     dxa_txbar_bus_if();
`endif

    // cta_table_if removed: cluster-contiguous LMEM placement lets the
    // DXA multicast writer compute receiver addresses as
    // `issuer_base + r × smem_stride`, eliminating the per-slot lookup table.

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_mem_if[`VX_CFG_NUM_LSU_BLOCKS]();

    // VX_lsu_scheduler instantiated per-LSU-block; all blocks have NUM_CLIENTS=2.
    // Block 0 wires client 1 to the warp-level TCU AGU; other blocks tie it off.
    // LSU client interfaces flow from execute as client 0.
    VX_lsu_sched_if lsu_client_if [`VX_CFG_NUM_LSU_BLOCKS]();
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_sched_empty;

`ifdef VX_CFG_TCU_META_ENABLE
    VX_lsu_sched_if tcu_mem_if();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH_BASE)
    ) mmu_dcache_if[DCACHE_NUM_REQS]();

    // VX_fetch -> VX_dcr_flush -> mmu_icache_if -> (i)MMU -> icache_bus_if.
    // The flush wrapper consumes the dcr-triggered cache-flush request and
    // injects a synthetic flush request into the icache stream, mirroring
    // the dcache path in VX_mem_unit.
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_FETCH_TAG_WIDTH)
    ) fetch_icache_if[1]();

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH_BASE)
    ) mmu_icache_if[1]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    localparam TCU_LMEM_BANK_ADDR_W = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE) - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);
    VX_mem_bus_if #(
        .DATA_SIZE  (`VX_CFG_LMEM_NUM_BANKS * LSU_WORD_SIZE),
        .TAG_WIDTH  (TCU_LMEM_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (TCU_LMEM_BANK_ADDR_W)
    ) tcu_lmem_if();
`endif

`ifdef PERF_ENABLE
    lmem_perf_t lmem_perf;
    coalescer_perf_t coalescer_perf;
    pipeline_perf_t pipeline_perf;
`ifdef VX_CFG_EXT_TCU_ENABLE
    tcu_perf_t tcu_perf;
    assign pipeline_perf.tcu = tcu_perf;
`endif
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.lmem = lmem_perf;
        sysmem_perf_tmp.coalescer = coalescer_perf;
    end
`endif

    VX_dcr_csr_if dcr_csr_if();

    // Single DCR-flush trigger from VX_dcr_data, fanned out below to BOTH
    // the dcache (via VX_mem_unit) and the icache (via the wrapper inserted
    // between VX_fetch and the iMMU). Without invalidating the icache too,
    // a kernel re-loaded to the same VMA after a CACHE_FLUSH would execute
    // stale lines from the previous launch.
    VX_dcr_flush_if dcr_flush_if();
    VX_dcr_flush_if dcr_flush_dcache_if();
    VX_dcr_flush_if dcr_flush_icache_if();

    assign dcr_flush_dcache_if.req = dcr_flush_if.req;
    assign dcr_flush_icache_if.req = dcr_flush_if.req;
    // Each VX_dcr_flush holds .done level-high until its req drops, so a
    // straight AND of the two dones reports the combined completion to
    // VX_dcr_data — which then drops req, re-arming both for the next flush.
    // When gfx extensions are enabled, also gate on the cluster-shared
    // gfx-cache flush done so that VX_dcr_data doesn't retire the flush DCR
    // before tcache/rcache/ocache have actually invalidated.
`ifdef EXT_GFX_ANY_ENABLE
    assign cluster_flush_if.req  = dcr_flush_if.req;
    assign dcr_flush_if.done = dcr_flush_dcache_if.done & dcr_flush_icache_if.done & cluster_flush_if.done;
`else
    assign dcr_flush_if.done = dcr_flush_dcache_if.done & dcr_flush_icache_if.done;
`endif

    wire dcr_busy;
    VX_dcr_data #(
        .INSTANCE_ID (`SFORMATF(("%s-dcr_data", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) dcr_data (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .dcr_csr_if (dcr_csr_if),
        .dcr_flush_if(dcr_flush_if),
        .dcr_busy   (dcr_busy)
    );

    `SCOPE_IO_SWITCH (3);

    wire sched_busy;
    VX_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-scheduler", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) scheduler (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sched_perf     (pipeline_perf.sched),
    `endif

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),

        .decode_sched_if(decode_sched_if),
        .issue_sched_if (issue_sched_if),
        .commit_sched_if(commit_sched_if),

        .kmu_bus_if     (kmu_bus_if),

        .schedule_if    (schedule_if),
        .sched_csr_if   (sched_csr_if),
        .gbar_bus_if    (gbar_bus_if),

        .busy           (sched_busy)
    );

    VX_fetch #(
        .INSTANCE_ID (`SFORMATF(("%s-fetch", INSTANCE_ID)))
    ) fetch (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .fetch_perf     (pipeline_perf.fetch),
    `endif
        .icache_bus_if  (fetch_icache_if[0]),
        .schedule_if    (schedule_if),
        .fetch_if       (fetch_if)
    );

    // Inject CACHE_FLUSH onto the icache stream so a host-side CMD_CACHE_FLUSH
    // (issued after every kernel launch) invalidates instruction-cache lines
    // belonging to the previous kernel image.
    VX_dcr_flush #(
        .WORD_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_FETCH_TAG_WIDTH)
    ) icache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (dcr_flush_icache_if),
        .core_bus_if  (fetch_icache_if[0]),
        .cache_bus_if (mmu_icache_if[0])
    );

    VX_decode #(
        .INSTANCE_ID (`SFORMATF(("%s-decode", INSTANCE_ID)))
    ) decode (
        .clk            (clk),
        .reset          (reset),
        .fetch_if       (fetch_if),
        .decode_if      (decode_if),
        .decode_sched_if(decode_sched_if)
    );

    VX_issue #(
        .INSTANCE_ID (`SFORMATF(("%s-issue", INSTANCE_ID)))
    ) issue (
        `SCOPE_IO_BIND  (1)

        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .issue_perf     (pipeline_perf.issue),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),
        .dispatch_if    (dispatch_if),
        .issue_sched_if (issue_sched_if)
    );

    VX_execute #(
        .INSTANCE_ID (`SFORMATF(("%s-execute", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) execute (
        `SCOPE_IO_BIND  (2)

        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf_tmp),
        .pipeline_perf  (pipeline_perf),
    `ifdef VX_CFG_EXT_TCU_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
    `endif

        // execute exposes LSU client interfaces; lsu_scheduler sits between
        // execute and lsu_mem_if (which connects to mem_unit).
        .lsu_client_if  (lsu_client_if),

    `ifdef VX_CFG_TCU_META_ENABLE
        .tcu_mem_if     (tcu_mem_if),
    `endif

        .dispatch_if    (dispatch_if),
        .commit_if      (commit_if),

        .sched_csr_if   (sched_csr_if),

        .dcr_csr_if     (dcr_csr_if),

    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_lmem_if    (tcu_lmem_if),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if (dxa_req_bus_if),
        .dxa_txbar_bus_if(dxa_txbar_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if     (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .om_bus_if      (om_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_bus_if  (raster_bus_if),
    `endif

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if)
    );

    VX_commit #(
        .INSTANCE_ID (`SFORMATF(("%s-commit", INSTANCE_ID)))
    ) commit (
        .clk            (clk),
        .reset          (reset),

        .commit_if      (commit_if),

        .writeback_if   (writeback_if),

        .commit_sched_if(commit_sched_if)
    );

    // Per-block VX_lsu_scheduler instances: all parameterized NUM_CLIENTS=2.
    // Block 0 wires client 1 to the warp-level TCU AGU; other blocks tie it off.
    // Symmetric NUM_CLIENTS keeps module generation uniform — tied-off clients
    // cost only a few muxes inside the round-robin arbiter.
`ifdef VX_CFG_TCU_META_ENABLE
    localparam LSU_SCHED_NUM_CLIENTS = 2;
`else
    localparam LSU_SCHED_NUM_CLIENTS = 1;
`endif
    for (genvar block_idx = 0; block_idx < `VX_CFG_NUM_LSU_BLOCKS; ++block_idx) begin : g_lsu_scheduler
        VX_lsu_sched_if sched_client_if [LSU_SCHED_NUM_CLIENTS]();

        // Client 0 — LSU (per-block). Forward the per-block lsu_client_if
        // master onto the scheduler's slave port via per-signal assigns
        // (interface arrays can't be wired with a single assign).
        assign sched_client_if[0].req_valid    = lsu_client_if[block_idx].req_valid;
        assign sched_client_if[0].req_data     = lsu_client_if[block_idx].req_data;
        assign lsu_client_if[block_idx].req_ready  = sched_client_if[0].req_ready;
        assign lsu_client_if[block_idx].rsp_valid  = sched_client_if[0].rsp_valid;
        assign lsu_client_if[block_idx].rsp_data   = sched_client_if[0].rsp_data;
        assign sched_client_if[0].rsp_ready    = lsu_client_if[block_idx].rsp_ready;

    `ifdef VX_CFG_TCU_META_ENABLE
        // Client 1 — TCU AGU on block 0; tied off on other blocks.
        if (block_idx == 0) begin : g_tcu_client
            assign sched_client_if[1].req_valid = tcu_mem_if.req_valid;
            assign sched_client_if[1].req_data  = tcu_mem_if.req_data;
            assign tcu_mem_if.req_ready         = sched_client_if[1].req_ready;
            assign tcu_mem_if.rsp_valid         = sched_client_if[1].rsp_valid;
            assign tcu_mem_if.rsp_data          = sched_client_if[1].rsp_data;
            assign sched_client_if[1].rsp_ready = tcu_mem_if.rsp_ready;
        end else begin : g_tcu_client_tieoff
            assign sched_client_if[1].req_valid = 1'b0;
            assign sched_client_if[1].req_data  = '0;
            assign sched_client_if[1].rsp_ready = 1'b1;
            `UNUSED_VAR (sched_client_if[1].req_ready)
            `UNUSED_VAR (sched_client_if[1].rsp_valid)
            `UNUSED_VAR (sched_client_if[1].rsp_data)
        end
    `endif

        VX_lsu_scheduler #(
            .INSTANCE_ID    (`SFORMATF(("%s-lsusched%0d", INSTANCE_ID, block_idx))),
            .NUM_CLIENTS    (LSU_SCHED_NUM_CLIENTS),
            .NUM_LANES      (`VX_CFG_NUM_LSU_LANES),
            .CORE_QUEUE_SIZE(`VX_CFG_LSUQ_IN_SIZE),
            .MEM_QUEUE_SIZE (`VX_CFG_LSUQ_OUT_SIZE)
        ) lsu_scheduler (
            .clk        (clk),
            .reset      (reset),
            .client_if  (sched_client_if),
            .empty      (lsu_sched_empty[block_idx]),
            .lsu_mem_if (lsu_mem_if[block_idx])
        );
    end

    VX_mem_unit #(
        .INSTANCE_ID (INSTANCE_ID)
    ) mem_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .lmem_perf     (lmem_perf),
        .coalescer_perf(coalescer_perf),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_lmem_if   (tcu_lmem_if),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_lmem_bus_if(dxa_lmem_bus_if),
        .dxa_txbar_bus_if(dxa_txbar_bus_if),
    `endif
        .lsu_mem_if    (lsu_mem_if),
        .dcr_flush_if  (dcr_flush_dcache_if),
        .dcache_bus_if (mmu_dcache_if)
    );

`ifdef VX_CFG_VM_ENABLE
`ifdef PERF_ENABLE
    mmu_perf_t dcache_mmu_perf;
    mmu_perf_t icache_mmu_perf;
    // Combine icache + dcache MMU counters into the pipeline_perf.mmu
    // struct so VX_csr_data can read them under MPM_CLASS_CORE.
    assign pipeline_perf.mmu.tlb_reads     = dcache_mmu_perf.tlb_reads     + icache_mmu_perf.tlb_reads;
    assign pipeline_perf.mmu.tlb_hits      = dcache_mmu_perf.tlb_hits      + icache_mmu_perf.tlb_hits;
    assign pipeline_perf.mmu.tlb_misses    = dcache_mmu_perf.tlb_misses    + icache_mmu_perf.tlb_misses;
    assign pipeline_perf.mmu.tlb_evictions = dcache_mmu_perf.tlb_evictions + icache_mmu_perf.tlb_evictions;
    assign pipeline_perf.mmu.ptw_walks     = dcache_mmu_perf.ptw_walks     + icache_mmu_perf.ptw_walks;
    assign pipeline_perf.mmu.ptw_latency   = dcache_mmu_perf.ptw_latency   + icache_mmu_perf.ptw_latency;
`endif

    // Per-core dcache MMU.
    VX_mmu #(
        .NUM_REQS  (DCACHE_NUM_REQS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH_BASE)
    ) dcache_mmu (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .mmu_perf      (dcache_mmu_perf),
    `endif
        .satp          (sched_csr_if.csr_satp),
        .lsu_mem_if    (mmu_dcache_if),
        .dcache_mem_if (dcache_bus_if)
    );

    // Per-core icache MMU. NUM_REQS=1.
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) icache_mmu_out_if[1]();

    VX_mmu #(
        .NUM_REQS  (1),
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH_BASE)
    ) icache_mmu (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .mmu_perf      (icache_mmu_perf),
    `endif
        .satp          (sched_csr_if.csr_satp),
        .lsu_mem_if    (mmu_icache_if),
        .dcache_mem_if (icache_mmu_out_if)
    );

    `ASSIGN_VX_MEM_BUS_IF (icache_bus_if, icache_mmu_out_if[0]);
`else
    // No-VM passthrough: same widths on both sides.
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_dcache_no_vm
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i], mmu_dcache_if[i]);
    end
    `ASSIGN_VX_MEM_BUS_IF (icache_bus_if, mmu_icache_if[0]);
`endif

    assign busy = sched_busy || dcr_busy || ~(&lsu_sched_empty);

    // BAR (vx_barrier / vx_barrier_arrive) drains LSU before suspending or registering arrival.
    assign warp_ctl_if.lsu_sched_drained = &lsu_sched_empty;

`ifdef PERF_ENABLE

    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rd_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_wr_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rsp_per_cycle;

    wire [1:0] perf_icache_pending_read_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)+1-1:0] perf_dcache_pending_read_cycle;

    reg [PERF_CTR_BITS-1:0] perf_icache_pending_reads;
    reg [PERF_CTR_BITS-1:0] perf_dcache_pending_reads;

    reg [PERF_CTR_BITS-1:0] perf_ifetches;
    reg [PERF_CTR_BITS-1:0] perf_loads;
    reg [PERF_CTR_BITS-1:0] perf_stores;

    wire perf_icache_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire perf_icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;

    wire [LSU_NUM_REQS-1:0] perf_dcache_rd_req_fire, perf_dcache_rd_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_wr_req_fire, perf_dcache_wr_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_rsp_fire;

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_perf_dcache
        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_j
            assign perf_dcache_rd_req_fire[i * `VX_CFG_NUM_LSU_LANES + j] = lsu_mem_if[i].req_valid && lsu_mem_if[i].req_data.mask[j] && lsu_mem_if[i].req_ready && ~lsu_mem_if[i].req_data.rw;
            assign perf_dcache_wr_req_fire[i * `VX_CFG_NUM_LSU_LANES + j] = lsu_mem_if[i].req_valid && lsu_mem_if[i].req_data.mask[j] && lsu_mem_if[i].req_ready && lsu_mem_if[i].req_data.rw;
            assign perf_dcache_rsp_fire[i * `VX_CFG_NUM_LSU_LANES + j] = lsu_mem_if[i].rsp_valid && lsu_mem_if[i].rsp_data.mask[j] && lsu_mem_if[i].rsp_ready;
        end
    end

    `BUFFER(perf_dcache_rd_req_fire_r, perf_dcache_rd_req_fire);
    `BUFFER(perf_dcache_wr_req_fire_r, perf_dcache_wr_req_fire);

    `POP_COUNT(perf_dcache_rd_req_per_cycle, perf_dcache_rd_req_fire_r);
    `POP_COUNT(perf_dcache_wr_req_per_cycle, perf_dcache_wr_req_fire_r);
    `POP_COUNT(perf_dcache_rsp_per_cycle, perf_dcache_rsp_fire);

    assign perf_icache_pending_read_cycle = perf_icache_req_fire - perf_icache_rsp_fire;
    assign perf_dcache_pending_read_cycle = perf_dcache_rd_req_per_cycle - perf_dcache_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_icache_pending_reads <= '0;
            perf_dcache_pending_reads <= '0;
        end else begin
            perf_icache_pending_reads <= $signed(perf_icache_pending_reads) + PERF_CTR_BITS'($signed(perf_icache_pending_read_cycle));
            perf_dcache_pending_reads <= $signed(perf_dcache_pending_reads) + PERF_CTR_BITS'($signed(perf_dcache_pending_read_cycle));
        end
    end

    reg [PERF_CTR_BITS-1:0] perf_icache_lat;
    reg [PERF_CTR_BITS-1:0] perf_dcache_lat;

    always @(posedge clk) begin
        if (reset) begin
            perf_ifetches   <= '0;
            perf_loads      <= '0;
            perf_stores     <= '0;
            perf_icache_lat <= '0;
            perf_dcache_lat <= '0;
        end else begin
            perf_ifetches   <= perf_ifetches   + PERF_CTR_BITS'(perf_icache_req_fire);
            perf_loads      <= perf_loads      + PERF_CTR_BITS'(perf_dcache_rd_req_per_cycle);
            perf_stores     <= perf_stores     + PERF_CTR_BITS'(perf_dcache_wr_req_per_cycle);
            perf_icache_lat <= perf_icache_lat + perf_icache_pending_reads;
            perf_dcache_lat <= perf_dcache_lat + perf_dcache_pending_reads;
        end
    end

    assign pipeline_perf.ifetches = perf_ifetches;
    assign pipeline_perf.loads = perf_loads;
    assign pipeline_perf.stores = perf_stores;
    assign pipeline_perf.ifetch_latency = perf_icache_lat;
    assign pipeline_perf.load_latency = perf_dcache_lat;

`endif

endmodule
