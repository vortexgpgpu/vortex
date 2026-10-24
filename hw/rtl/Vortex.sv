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

module Vortex import VX_gpu_pkg::*, VX_trace_pkg::*, VX_tlb_pkg::*; (
    `SCOPE_IO_DECL

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid [VX_MEM_PORTS],
    output wire                             mem_req_rw [VX_MEM_PORTS],
    output wire [VX_MEM_BYTEEN_WIDTH-1:0]   mem_req_byteen [VX_MEM_PORTS],
    output wire [VX_MEM_ADDR_WIDTH-1:0]     mem_req_addr [VX_MEM_PORTS],
    output wire [VX_MEM_DATA_WIDTH-1:0]     mem_req_data [VX_MEM_PORTS],
    output wire [VX_MEM_TAG_WIDTH-1:0]      mem_req_tag [VX_MEM_PORTS],
    input  wire                             mem_req_ready [VX_MEM_PORTS],

    // Memory response
    input wire                              mem_rsp_valid [VX_MEM_PORTS],
    input wire [VX_MEM_DATA_WIDTH-1:0]      mem_rsp_data [VX_MEM_PORTS],
    input wire [VX_MEM_TAG_WIDTH-1:0]       mem_rsp_tag [VX_MEM_PORTS],
    output wire                             mem_rsp_ready [VX_MEM_PORTS],

    // DCR write request
    input  wire                             dcr_req_valid,
    input  wire                             dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_req_data,

    // DCR read response
    output wire                             dcr_rsp_valid,
    output wire [VX_DCR_DATA_WIDTH-1:0]     dcr_rsp_data,

    // Command-processor DMA into the last-level cache (VX_CFG_CP_ON_LLC)
    input  wire                             cp_mem_req_valid,
    input  wire                             cp_mem_req_rw,
    input  wire                             cp_mem_req_noalloc,
    input  wire [CP_MEM_ADDR_WIDTH-1:0]     cp_mem_req_addr,
    input  wire [CP_MEM_DATA_SIZE*8-1:0]    cp_mem_req_data,
    input  wire [CP_MEM_DATA_SIZE-1:0]      cp_mem_req_byteen,
    input  wire [CP_MEM_TAG_WIDTH-1:0]      cp_mem_req_tag,
    output wire                             cp_mem_req_ready,

    output wire                             cp_mem_rsp_valid,
    output wire [CP_MEM_DATA_SIZE*8-1:0]    cp_mem_rsp_data,
    output wire [CP_MEM_TAG_WIDTH-1:0]      cp_mem_rsp_tag,
    input  wire                             cp_mem_rsp_ready,

    // ctrl/status
    input  wire                             start,
    output wire                             busy
);
    `STATIC_ASSERT((`VX_CFG_NUM_CORES % `VX_CFG_SOCKET_SIZE) == 0, ("NUM_CORES must be a multiple of SOCKET_SIZE"));

    // A write-back LLC must be the only instance: replicated ones never make
    // their dirty lines visible to each other.
    `STATIC_ASSERT(!(DCACHE_IS_LLC && `VX_CFG_DCACHE_WRITEBACK) || ((`VX_CFG_NUM_CORES == 1) && (`VX_CFG_NUM_CLUSTERS == 1)),
        ("a write-back dcache that is the last-level cache must be the only dcache"));
    `STATIC_ASSERT(!(L2_IS_LLC && `VX_CFG_L2_WRITEBACK) || (`VX_CFG_NUM_CLUSTERS == 1),
        ("a write-back L2 that is the last-level cache must be the only L2"));

    // Every cache strictly above the LLC must be write-through.
    // A WB intermediate could absorb a hart-B store without the LLC
    // seeing it; a later SC from hart-A on the same line would
    // spuriously succeed. RVA permits spurious failure, not spurious success.
`ifdef VX_CFG_EXT_A_ENABLE
  `ifdef VX_CFG_L3_ENABLE
    `STATIC_ASSERT(`VX_CFG_DCACHE_WRITEBACK == 0, ("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L3 is the LLC"));
    `STATIC_ASSERT(`VX_CFG_L2_WRITEBACK == 0,     ("AMO requires write-through L2 (L2_WRITEBACK=0) when L3 is the LLC"));
  `elsif VX_CFG_L2_ENABLE
    `STATIC_ASSERT(`VX_CFG_DCACHE_WRITEBACK == 0, ("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L2 is the LLC"));
  `endif
`endif

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.req_valid = dcr_req_valid;
    assign dcr_bus_if.req_data.rw = dcr_req_rw;
    assign dcr_bus_if.req_data.addr = dcr_req_addr;
    assign dcr_bus_if.req_data.data = dcr_req_data;
    assign dcr_rsp_valid = dcr_bus_if.rsp_valid;
    assign dcr_rsp_data = dcr_bus_if.rsp_data;

    wire kmu_start;
    wire cp_mem_busy;

    // Kernel Management Unit
    VX_kmu_bus_if kmu_bus_in[1]();
    wire kmu_busy;
`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_launch_if raster_launch_if();
`endif
    VX_kmu #(
        .INSTANCE_ID ("kmu")
    ) kmu (
        .clk        (clk),
        .reset      (reset),
        .start      (kmu_start),
        .busy       (kmu_busy),
        .dcr_req_valid (dcr_req_valid),
        .dcr_req_rw (dcr_req_rw),
        .dcr_req_addr(dcr_req_addr),
        .dcr_req_data(dcr_req_data),
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_launch_if (raster_launch_if),
    `endif
        .kmu_bus_if (kmu_bus_in[0])
    );

`ifdef SCOPE
    localparam scope_cluster = 0;
    `SCOPE_IO_SWITCH (`VX_CFG_NUM_CLUSTERS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l3_perf;
    mem_perf_t mem_perf;
    sysmem_perf_t sysmem_perf;
    always @(*) begin
        sysmem_perf = '0;
        sysmem_perf.l3cache = l3_perf;
        sysmem_perf.mem = mem_perf;
    end
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (L2_SECTOR_SIZE),
        .TAG_WIDTH (L3_TAG_WIDTH)
    ) per_cluster_mem_bus_if[L3_CACHE_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (L3_SECTOR_SIZE),
        .TAG_WIDTH (L3_MEM_TAG_WIDTH)
    ) mem_bus_if[L3_MEM_PORTS]();
    // no request held in the L3 bypass path short of the memory ports
    wire l3_req_empty;

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`VX_CFG_L3_SIZE),
        .LINE_SIZE      (`VX_CFG_L3_LINE_SIZE),
        .SECTOR_SIZE    (L3_SECTOR_SIZE),
        .NUM_BANKS      (L3_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_L3_NUM_WAYS),
        .WORD_SIZE      (L3_WORD_SIZE),
        .NUM_REQS       (L3_CACHE_REQS),
        .MEM_PORTS      (L3_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_L3_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_L3_MSHR_SIZE),
        .MSHR_LINES     (`VX_CFG_L3_MSHR_LINES),
        .MRSQ_SIZE      (`VX_CFG_L3_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_L3_MREQ_SIZE),
        .LATENCY        (`VX_CFG_L3_LATENCY),
        .TAG_WIDTH      (L3_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_L3_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_L3_DIRTYBYTES),
        .LAZY_FETCH     (`VX_CFG_L3_LAZY_FETCH),
        .WRITE_AROUND   (CP_ON_L3),
        .REPL_POLICY    (`VX_CFG_L3_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`VX_CFG_L3_ENABLED),
        .IS_LLC         (L3_IS_LLC),
        .SHARED_LLC     (L3_SHARED_LLC),
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) l3cache (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .cache_perf     (l3_perf),
    `endif

        .core_bus_if    (per_cluster_mem_bus_if),
        .mem_bus_if     (mem_bus_if),
        .req_empty      (l3_req_empty)
    );

    for (genvar i = 0; i < L3_MEM_PORTS; ++i) begin : g_mem_bus_if
        assign mem_req_valid[i]  = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]     = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i] = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i]   = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i]   = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i]    = mem_bus_if[i].req_data.tag;
        `UNUSED_VAR (mem_bus_if[i].req_data.attr)
        assign mem_bus_if[i].req_ready = mem_req_ready[i];

        assign mem_bus_if[i].rsp_valid     = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag  = mem_rsp_tag[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

    // Command-processor DMA: one leg per cluster so every instance owns its
    // port's drivers; only cluster 0's leg carries traffic, and only under
    // CP_ON_L2. Under CP_ON_L3 the DMA is the L3's last client.
    VX_mem_bus_if #(
        .DATA_SIZE (CP_MEM_DATA_SIZE),
        .TAG_WIDTH (CP_SLOT_TAG_WIDTH)
    ) per_cluster_cp_mem_if [`VX_CFG_NUM_CLUSTERS] ();

    if ((CP_ON_L2 != 0) || (CP_ON_L3 != 0)) begin : g_cp_mem
        VX_mem_bus_if #(
            .DATA_SIZE (CP_MEM_DATA_SIZE),
            .TAG_WIDTH (CP_SLOT_TAG_WIDTH)
        ) cp_mem_if ();

        VX_cp_llc_port #(
            .DATA_SIZE      (CP_MEM_DATA_SIZE),
            .SLOT_TAG_WIDTH (CP_SLOT_TAG_WIDTH),
            .TAG_WIDTH      (CP_MEM_TAG_WIDTH),
            .ADDR_WIDTH     (CP_MEM_ADDR_WIDTH)
        ) cp_llc_port (
            .clk         (clk),
            .reset       (reset),
            .req_valid   (cp_mem_req_valid),
            .req_rw      (cp_mem_req_rw),
            .req_noalloc (cp_mem_req_noalloc),
            .req_addr    (cp_mem_req_addr),
            .req_data    (cp_mem_req_data),
            .req_byteen  (cp_mem_req_byteen),
            .req_tag     (cp_mem_req_tag),
            .req_ready   (cp_mem_req_ready),
            .rsp_valid   (cp_mem_rsp_valid),
            .rsp_data    (cp_mem_rsp_data),
            .rsp_tag     (cp_mem_rsp_tag),
            .rsp_ready   (cp_mem_rsp_ready),
            .start_in    (start),
            .start_out   (kmu_start),
            .busy        (cp_mem_busy),
            .mem_bus_if  (cp_mem_if)
        );

        if (CP_ON_L2 != 0) begin : g_l2
            `ASSIGN_VX_MEM_BUS_IF (per_cluster_cp_mem_if[0], cp_mem_if);
        end else begin : g_l3
            `ASSIGN_VX_MEM_BUS_IF (per_cluster_mem_bus_if[L3_CP_IDX], cp_mem_if);
            `INIT_VX_MEM_BUS_IF (per_cluster_cp_mem_if[0])
        end
        for (genvar c = 1; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_cp_mem_legs
            `INIT_VX_MEM_BUS_IF (per_cluster_cp_mem_if[c])
        end
    end else begin : g_no_cp_mem
        assign cp_mem_req_ready = 1'b0;
        assign cp_mem_rsp_valid = 1'b0;
        assign cp_mem_rsp_data  = '0;
        assign cp_mem_rsp_tag   = '0;
        `UNUSED_VAR ({cp_mem_req_valid, cp_mem_req_rw, cp_mem_req_noalloc, cp_mem_req_addr, cp_mem_req_data, cp_mem_req_byteen, cp_mem_req_tag, cp_mem_rsp_ready})
        assign kmu_start   = start;
        assign cp_mem_busy = 1'b0;
        for (genvar c = 0; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_cp_mem_legs
            `INIT_VX_MEM_BUS_IF (per_cluster_cp_mem_if[c])
        end
    end

    wire [`VX_CFG_NUM_CLUSTERS-1:0] per_cluster_busy;

    VX_kmu_bus_if per_cluster_kmu_bus_if[`VX_CFG_NUM_CLUSTERS]();

    VX_kmu_bus_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_CLUSTERS),
        .DEST_DIV   (KMU_DEST_DIV_DEVICE),
        .OUT_BUF    ((`VX_CFG_NUM_CLUSTERS > 1) ? 3 : 0)  // register per-cluster kmu fan-out (SLR-crossing skid)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_bus_in),
        .bus_out_if (per_cluster_kmu_bus_if)
    );

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_launch_if per_cluster_raster_launch_if[`VX_CFG_NUM_CLUSTERS]();
    VX_raster_launch_fork #(
        .NUM_OUTPUTS (`VX_CFG_NUM_CLUSTERS)
    ) raster_launch_fork (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (raster_launch_if),
        .bus_out_if (per_cluster_raster_launch_if)
    );
`endif

    // The device MMU surface filters the DCR stream: it assembles the
    // page-table root, pulses the TLB flush, and answers fault reads before
    // the rest of the DCR traffic fans to the clusters.
    VX_dcr_bus_if dcr_cluster_src_if();
`ifdef VX_CFG_VM_ENABLE
    wire [`VX_CFG_XLEN-1:0] mmu_satp;
    wire                    mmu_flush_req;
    wire [`VX_CFG_NUM_CLUSTERS-1:0] cl_mmu_flush_done;

    // Per-cluster L2-TLB miss export buses (used by the device-level walker).
    VX_tlb_bus_if #(.ID_WIDTH (L2_TLB_SLOT_WIDTH)) per_cluster_dev_ptw_if [`VX_CFG_NUM_CLUSTERS] ();

    // One shared walker at the device: the clusters' L2-TLB miss buses arb
    // into it, and the flush done-tree gains the walker's leg. Structural
    // faults surface here.
    VX_tlb_bus_if #(.ID_WIDTH (TLB_DEV_ID_WIDTH)) dev_ptw_bus_if ();

    VX_tlb_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_CLUSTERS),
        .ID_WIDTH_IN (L2_TLB_SLOT_WIDTH),
        .OUT_BUF     (3)
    ) tlb_dev_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_cluster_dev_ptw_if),
        .bus_out_if (dev_ptw_bus_if)
    );

    VX_tlb_flush_if dev_ptw_flush_if ();
    VX_mmu_fault_if dev_ptw_fault_if ();
    wire dev_ptw_empty;

    // PTE fetches ride the LLC: the L3 client slot normally, or the single
    // cluster's L2 when that L2 is the LLC (PTW_ON_L2).
    localparam PTW_MEM_DATA_SIZE = (PTW_ON_L2 != 0) ? `VX_CFG_L1_LINE_SIZE : L2_SECTOR_SIZE;
    localparam PTW_MEM_TAG_WIDTH = (PTW_ON_L2 != 0) ? L2_TAG_WIDTH : L3_TAG_WIDTH;

    VX_mem_bus_if #(
        .DATA_SIZE (PTW_MEM_DATA_SIZE),
        .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
    ) dev_ptw_mem_if ();

    // One leg per cluster so every instance owns its port's drivers; only
    // cluster 0's leg carries traffic, and only under PTW_ON_L2.
    VX_mem_bus_if #(
        .DATA_SIZE (PTW_MEM_DATA_SIZE),
        .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
    ) per_cluster_ptw_mem_if [`VX_CFG_NUM_CLUSTERS] ();

    VX_ptw #(
        .ID_WIDTH       (TLB_DEV_ID_WIDTH),
        .DATA_SIZE      (PTW_MEM_DATA_SIZE),
        .MEM_TAG_WIDTH  (PTW_MEM_TAG_WIDTH)
    ) dev_ptw (
        .clk        (clk),
        .reset      (reset),
        .satp       (mmu_satp),
        .miss_if    (dev_ptw_bus_if),
        .mem_bus_if (dev_ptw_mem_if),
        .flush_if   (dev_ptw_flush_if),
        .fault_if   (dev_ptw_fault_if),
        .empty      (dev_ptw_empty)
    );

    if (PTW_ON_L2 == 0) begin : g_ptw_llc_client
        `ASSIGN_VX_MEM_BUS_IF (per_cluster_mem_bus_if[L3_PTW_IDX], dev_ptw_mem_if);
        for (genvar c = 0; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_ptw_mem_legs
            `INIT_VX_MEM_BUS_IF (per_cluster_ptw_mem_if[c])
        end
    end else begin : g_ptw_l2_client
        `ASSIGN_VX_MEM_BUS_IF (per_cluster_ptw_mem_if[0], dev_ptw_mem_if);
    end

    assign dev_ptw_flush_if.req = mmu_flush_req;

    // The walker is the only fault source and owns its own legs on the DCR
    // surface; the clusters report translation drains only.
    VX_mmu_dcr mmu_dcr (
        .clk                (clk),
        .reset              (reset),
        .dcr_bus_if         (dcr_bus_if),
        .dcr_bus_out_if     (dcr_cluster_src_if),
        .satp               (mmu_satp),
        .flush_req          (mmu_flush_req),
        .cluster_flush_done (cl_mmu_flush_done),
        .walker_flush_done  (dev_ptw_flush_if.done),
        .fault_valid        (dev_ptw_fault_if.valid),
        .fault_va           (dev_ptw_fault_if.va),
        .fault_access       (dev_ptw_fault_if.access),
        .fault_amo          (dev_ptw_fault_if.amo)
    );

    wire dev_mmu_busy = ~dev_ptw_empty;
`else
    assign dcr_cluster_src_if.req_valid = dcr_bus_if.req_valid;
    assign dcr_cluster_src_if.req_data  = dcr_bus_if.req_data;
    assign dcr_bus_if.rsp_valid = dcr_cluster_src_if.rsp_valid;
    assign dcr_bus_if.rsp_data  = dcr_cluster_src_if.rsp_data;

    wire dev_mmu_busy = 1'b0;
`endif

    VX_dcr_bus_if per_cluster_dcr_bus_if[`VX_CFG_NUM_CLUSTERS]();
    VX_dcr_arb #(
        .NUM_REQS    (`VX_CFG_NUM_CLUSTERS),
        .REQ_OUT_BUF ((`VX_CFG_NUM_CLUSTERS > 1) ? 1 : 0)
    ) dcr_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_cluster_src_if),
        .bus_out_if (per_cluster_dcr_bus_if)
    );

    // A global barrier's participants can span clusters, so its arrivals are
    // counted once for the whole device.
    VX_gbar_bus_if per_cluster_gbar_bus_if[`VX_CFG_NUM_CLUSTERS]();
    VX_gbar_bus_if gbar_bus_if();

    VX_gbar_arb #(
        .NUM_REQS    (`VX_CFG_NUM_CLUSTERS),
        .REQ_OUT_BUF ((`VX_CFG_NUM_CLUSTERS > 1) ? 3 : 0),
        .RSP_OUT_BUF ((`VX_CFG_NUM_CLUSTERS > 1) ? 3 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_cluster_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID ("gbar")
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if)
    );

    // Generate all clusters
    for (genvar cluster_id = 0; cluster_id < `VX_CFG_NUM_CLUSTERS; ++cluster_id) begin : g_clusters

        VX_cluster #(
            .INSTANCE_ID (`SFORMATF(("cluster%0d", cluster_id)))
        ) cluster (
            `SCOPE_IO_BIND (scope_cluster + cluster_id)

            .clk                (clk),
            .reset              (reset),
            .cluster_id         (CLUSTER_ID_WIDTH'(cluster_id)),

        `ifdef PERF_ENABLE
            .sysmem_perf        (sysmem_perf),
        `endif

            .dcr_bus_if         (per_cluster_dcr_bus_if[cluster_id]),

            .mem_bus_if         (per_cluster_mem_bus_if[cluster_id * L2_MEM_PORTS +: L2_MEM_PORTS]),

            .kmu_bus_if         (per_cluster_kmu_bus_if[cluster_id +: 1]),

            .gbar_bus_if        (per_cluster_gbar_bus_if[cluster_id]),

        `ifdef VX_CFG_EXT_RASTER_ENABLE
            .raster_launch_if   (per_cluster_raster_launch_if[cluster_id +: 1]),
        `endif

        `ifdef VX_CFG_VM_ENABLE
            .dev_ptw_if         (per_cluster_dev_ptw_if[cluster_id]),
            .dev_ptw_mem_if     (per_cluster_ptw_mem_if[cluster_id]),
            .mmu_flush_req      (mmu_flush_req),
            .mmu_flush_done     (cl_mmu_flush_done[cluster_id]),
        `endif

            .cp_mem_if          (per_cluster_cp_mem_if[cluster_id]),

            .busy               (per_cluster_busy[cluster_id])
        );
    end
    // Launch liveness: a beat resident in a per-cluster output skid is folded into
    // busy combinationally. The device input
    // (kmu_bus_in) needs no separate term -- kmu_busy (combinational) covers the
    // presented cycle and the registered per_cluster_busy covers the cycles after.
    wire [`VX_CFG_NUM_CLUSTERS-1:0] per_cluster_kmu_valid;
    for (genvar c = 0; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_kmu_link_valid
        assign per_cluster_kmu_valid[c] = per_cluster_kmu_bus_if[c].valid;
    end
    wire busy_r;
    `BUFFER_EX(busy_r, kmu_busy | dcr_bus_if.req_valid | (|per_cluster_busy) | dev_mmu_busy | ~l3_req_empty, 1'b1, 1, (`VX_CFG_NUM_CLUSTERS > 1));
    assign busy = busy_r | kmu_busy | dcr_bus_if.req_valid | (|per_cluster_kmu_valid) | cp_mem_busy;

`ifdef PERF_ENABLE

    localparam MEM_PORTS_CTR_W = `CLOG2(VX_MEM_PORTS+1);

    wire [VX_MEM_PORTS-1:0] mem_req_fire, mem_rsp_fire;
    wire [VX_MEM_PORTS-1:0] mem_rd_req_fire, mem_wr_req_fire;

    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_perf_ctrs
        assign mem_req_fire[i] = mem_req_valid[i] & mem_req_ready[i];
        assign mem_rsp_fire[i] = mem_rsp_valid[i] & mem_rsp_ready[i];
        assign mem_rd_req_fire[i] = mem_req_fire[i] & ~mem_req_rw[i];
        assign mem_wr_req_fire[i] = mem_req_fire[i] & mem_req_rw[i];
    end

    wire [MEM_PORTS_CTR_W-1:0] perf_mem_reads_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_writes_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_rsps_per_cycle;

    `POP_COUNT(perf_mem_reads_per_cycle, mem_rd_req_fire);
    `POP_COUNT(perf_mem_writes_per_cycle, mem_wr_req_fire);
    `POP_COUNT(perf_mem_rsps_per_cycle, mem_rsp_fire);

    reg [PERF_CTR_BITS-1:0] perf_mem_pending_reads;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= '0;
        end else begin
            perf_mem_pending_reads <= $signed(perf_mem_pending_reads) +
                PERF_CTR_BITS'($signed((MEM_PORTS_CTR_W+1)'(perf_mem_reads_per_cycle) - (MEM_PORTS_CTR_W+1)'(perf_mem_rsps_per_cycle)));
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            mem_perf <= '0;
        end else begin
            mem_perf.reads <= mem_perf.reads + PERF_CTR_BITS'(perf_mem_reads_per_cycle);
            mem_perf.writes <= mem_perf.writes + PERF_CTR_BITS'(perf_mem_writes_per_cycle);
            mem_perf.latency <= mem_perf.latency + perf_mem_pending_reads;
        end
    end

`endif

    // dump device configuration
    initial begin
        `TRACE(0, ("CONFIGS: num_threads=%0d, num_warps=%0d, num_cores=%0d, num_clusters=%0d, socket_size=%0d, local_mem_base=0x%0h, num_barriers=%0d\n",
                    `VX_CFG_NUM_THREADS, `VX_CFG_NUM_WARPS, `VX_CFG_NUM_CORES, `VX_CFG_NUM_CLUSTERS, `VX_CFG_SOCKET_SIZE, `VX_MEM_LMEM_BASE_ADDR, `VX_CFG_NUM_BARRIERS))
    end

`ifdef DBG_TRACE_MEM
    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_trace
        always @(posedge clk) begin
            if (mem_bus_if[i].req_valid && mem_bus_if[i].req_ready) begin
                if (mem_bus_if[i].req_data.rw) begin
                    `TRACE(2, ("%t: MEM Wr Req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.data, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end else begin
                    `TRACE(2, ("%t: MEM Rd Req[%0d]: addr=0x%0h, byteen=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end
            end
            if (mem_bus_if[i].rsp_valid && mem_bus_if[i].rsp_ready) begin
                `TRACE(2, ("%t: MEM Rd Rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n", $time, i, mem_bus_if[i].rsp_data.data, mem_bus_if[i].rsp_data.tag.value, mem_bus_if[i].rsp_data.tag.uuid))
            end
        end
    end
`endif

`ifdef SIMULATION
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule
