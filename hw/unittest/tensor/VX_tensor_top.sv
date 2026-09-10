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

// Standalone tensor-subsystem synthesis sandbox.
//
// Reproduces one socket's tensor-compute slice exactly as VX_core/VX_socket
// wire it, with decode/issue/ALU/FPU and graphics removed:
//
//   per core (VX_CFG_SOCKET_SIZE):
//     VX_tcu_unit      : tensor-core unit (dispatch/commit from top ports)
//     VX_lsu_scheduler : per-LSU-block, 2 clients — the ordinary LSU stream
//                        (top ports) and the TCU AGU metadata-load port
//     VX_mem_unit      : LMEM + coalescer + LMEM-DMA arb (TCU WGMMA + DXA
//                        staging converge here) + dcache adapters
//   per socket:
//     VX_dxa_core       : descriptor-driven tile-transfer engine (DCR-fed)
//     VX_mem_bus_switch : DXA staging writes back to the owning core's LMEM
//     VX_cache_cluster  : the real L1 dcache
//     VX_mem_bus_arb    : the socket memory arbiter (icache slot tied off,
//                         dcache + DXA gmem live) -> shared L2
//   cluster tail:
//     VX_cache_wrap L2  : PASSTHRU when VX_CFG_L2_ENABLE is off
//
// Memory hierarchy: TCU WGMMA prefetches and DXA staging writes meet at the
// real LMEM DMA arbiter inside VX_mem_unit; TCU_LD metadata loads merge with
// the ordinary LSU stream at block-0's VX_lsu_scheduler; global traffic
// (dcache refills, DXA tile reads) contends at the socket arbiter and the
// shared L2, whose memory side exits to top-level ports.
//
// Fidelity: every tensor timing boundary is present and loaded by its real
// neighbour. What is removed (decode/issue, scalar pipelines, icache) never
// drove a tensor boundary net. All widths follow the production VX_gpu_pkg
// localparams so structure matches the real RTL.
//
// No component may be optimized out: every DUT input is driven by a
// top-level port and every DUT output is observed at a top-level port.

module VX_tensor_top import VX_gpu_pkg::*, VX_tcu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "tensor_top",
    // Derived port geometry (do not override).
    parameter NC            = `VX_CFG_SOCKET_SIZE,
    parameter NB            = `VX_CFG_ISSUE_WIDTH,
    parameter NUM_LSU_BLKS  = `VX_CFG_NUM_LSU_BLOCKS,
    parameter LSU_STREAMS   = NC * `VX_CFG_NUM_LSU_BLOCKS,
    parameter DC_MEM_PORTS  = L1_MEM_PORTS,
    parameter L2_MEM_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(L2_SECTOR_SIZE)
) (
    input  wire clk,
    input  wire reset,

    // -----------------------------------------------------------------------
    // TCU dispatch/commit streams, flattened [core][issue-slot]
    // -----------------------------------------------------------------------
    input  wire [NC-1:0][NB-1:0]        dispatch_valid,
    input  dispatch_t [NC-1:0][NB-1:0]  dispatch_data,
    output wire [NC-1:0][NB-1:0]        dispatch_ready,

    output wire [NC-1:0][NB-1:0]        commit_valid,
    output commit_t [NC-1:0][NB-1:0]    commit_data,
    input  wire [NC-1:0][NB-1:0]        commit_ready,

    // -----------------------------------------------------------------------
    // Ordinary LSU client streams, flattened [core][lsu-block]
    // (client 0 of each VX_lsu_scheduler; the TCU AGU is client 1 on block 0)
    // -----------------------------------------------------------------------
    input  wire [LSU_STREAMS-1:0]           lsu_req_valid,
    input  lsu_req_data_t [LSU_STREAMS-1:0] lsu_req_data,
    output wire [LSU_STREAMS-1:0]           lsu_req_ready,

    output wire [LSU_STREAMS-1:0]           lsu_rsp_valid,
    output lsu_rsp_data_t [LSU_STREAMS-1:0] lsu_rsp_data,
    input  wire [LSU_STREAMS-1:0]           lsu_rsp_ready,

    // -----------------------------------------------------------------------
    // DXA programming + per-core request streams
    // -----------------------------------------------------------------------
    input  wire                         dcr_req_valid,   // descriptor-table writes
    input  dcr_req_t                    dcr_req_data,
    output wire                         dcr_rsp_valid,   // observed: keeps the table live

    input  wire [NC-1:0]                dxa_req_valid,
    input  dxa_req_data_t [NC-1:0]      dxa_req_data,
    output wire [NC-1:0]                dxa_req_ready,
    output wire                         dxa_busy,

    // DXA completion -> barrier release, observed per core
    output wire [NC-1:0]                txbar_valid,
    output txbar_t [NC-1:0]             txbar_data,
    input  wire [NC-1:0]                txbar_ready,

    // -----------------------------------------------------------------------
    // L2 memory side (master): dcache refills + DXA tile reads exit here
    // -----------------------------------------------------------------------
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_req_valid,
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_req_rw,
    output wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE-1:0]   l2_mem_req_byteen,
    output wire [L2_MEM_PORTS-1:0][L2_MEM_ADDR_W-1:0]    l2_mem_req_addr,
    output wire [L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0] l2_mem_req_tag,
    output wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE*8-1:0] l2_mem_req_data,
    input  wire [L2_MEM_PORTS-1:0]                       l2_mem_req_ready,

    input  wire [L2_MEM_PORTS-1:0]                       l2_mem_rsp_valid,
    input  wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE*8-1:0] l2_mem_rsp_data,
    input  wire [L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0] l2_mem_rsp_tag,
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_rsp_ready,

    output wire empty
);
    localparam TCU_LMEM_BANK_ADDR_W = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE) - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);

`ifdef TCU_META_ENABLE
    localparam LSU_SCHED_NUM_CLIENTS = 2;
`else
    localparam LSU_SCHED_NUM_CLIENTS = 1;
`endif

    // ------------------------------------------------------------------------
    // Per-core DXA staging returns + collected dcache core ports
    // ------------------------------------------------------------------------
    VX_dxa_req_bus_if per_core_dxa_req_bus_if[NC]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) per_core_dxa_lmem_bus_if[NC]();

    // Every core's dcache ports, collected for the shared L1 dcache cluster.
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) all_dc_core_if [NC * DCACHE_NUM_REQS] ();

    wire [NC-1:0] mem_unit_empty;
    wire [NC-1:0] lsu_sched_empty;

    // ========================================================================
    // Per-core tensor slice: TCU + LSU schedulers + mem_unit
    // ========================================================================
    for (genvar c = 0; c < NC; ++c) begin : g_cores

        VX_dispatch_if dispatch_if[NB]();
        VX_commit_if   commit_if[NB]();

        for (genvar b = 0; b < NB; ++b) begin : g_dispatch_bind
            assign dispatch_if[b].valid   = dispatch_valid[c][b];
            assign dispatch_if[b].data    = dispatch_data[c][b];
            assign dispatch_ready[c][b]   = dispatch_if[b].ready;
            assign commit_valid[c][b]     = commit_if[b].valid;
            assign commit_data[c][b]      = commit_if[b].data;
            assign commit_if[b].ready     = commit_ready[c][b];
        end

        VX_mem_bus_if #(
            .DATA_SIZE  (`VX_CFG_LMEM_NUM_BANKS * LSU_WORD_SIZE),
            .TAG_WIDTH  (TCU_LMEM_TAG_W),
            .ATTR_WIDTH (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH (TCU_LMEM_BANK_ADDR_W)
        ) tcu_lmem_if();

    `ifdef TCU_META_ENABLE
        VX_lsu_sched_if tcu_mem_if();
    `endif

        VX_tcu_unit #(
            .INSTANCE_ID (`SFORMATF(("%s-tcu%0d", INSTANCE_ID, c)))
        ) tcu_unit (
            `SCOPE_IO_BIND (c)
            .clk         (clk),
            .reset       (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .tcu_lmem_if (tcu_lmem_if),
        `endif
        `ifdef TCU_META_ENABLE
            .tcu_mem_if  (tcu_mem_if),
        `endif
            .dispatch_if (dispatch_if),
            .commit_if   (commit_if)
        );

        // --- LSU schedulers: client 0 from top ports, client 1 = TCU AGU ---
        VX_lsu_mem_if #(
            .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lsu_mem_if[NUM_LSU_BLKS]();

        wire [NUM_LSU_BLKS-1:0] block_sched_empty;

        for (genvar block_idx = 0; block_idx < NUM_LSU_BLKS; ++block_idx) begin : g_lsu_scheduler
            localparam s = c * NUM_LSU_BLKS + block_idx;

            VX_lsu_sched_if sched_client_if [LSU_SCHED_NUM_CLIENTS]();

            assign sched_client_if[0].req_valid = lsu_req_valid[s];
            assign sched_client_if[0].req_data  = lsu_req_data[s];
            assign lsu_req_ready[s]             = sched_client_if[0].req_ready;
            assign lsu_rsp_valid[s]             = sched_client_if[0].rsp_valid;
            assign lsu_rsp_data[s]              = sched_client_if[0].rsp_data;
            assign sched_client_if[0].rsp_ready = lsu_rsp_ready[s];

        `ifdef TCU_META_ENABLE
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
                .INSTANCE_ID    (`SFORMATF(("%s-lsusched%0d-%0d", INSTANCE_ID, c, block_idx))),
                .NUM_CLIENTS    (LSU_SCHED_NUM_CLIENTS),
                .NUM_LANES      (`VX_CFG_NUM_LSU_LANES),
                .CORE_QUEUE_SIZE(`VX_CFG_LSU_QUEUE_IN_SIZE),
                .MEM_QUEUE_SIZE (LSU_QUEUE_OUT_SIZE),
                .PENDING_SIZE   (`VX_CFG_LSU_PENDING_SIZE)
            ) lsu_scheduler (
                .clk        (clk),
                .reset      (reset),
                .client_if  (sched_client_if),
                .empty      (block_sched_empty[block_idx]),
                .lsu_mem_if (lsu_mem_if[block_idx])
            );
        end

        assign lsu_sched_empty[c] = & block_sched_empty;

        // --- mem_unit: LMEM + DMA arb + dcache adapters ---
        VX_txbar_bus_if dxa_txbar_bus_if();
        assign txbar_valid[c]           = dxa_txbar_bus_if.valid;
        assign txbar_data[c]            = dxa_txbar_bus_if.data;
        assign dxa_txbar_bus_if.ready   = txbar_ready[c];

        VX_dcr_flush_if dcr_flush_dcache_if();
        assign dcr_flush_dcache_if.req = 1'b0;
        `UNUSED_VAR (dcr_flush_dcache_if.done)

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH_BASE)
        ) dcache_bus_if[DCACHE_NUM_REQS]();

        VX_mem_unit #(
            .INSTANCE_ID (`SFORMATF(("%s-core%0d", INSTANCE_ID, c)))
        ) mem_unit (
            .clk             (clk),
            .reset           (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .tcu_lmem_if     (tcu_lmem_if),
        `endif
        `ifdef VX_CFG_EXT_DXA_ENABLE
            .dxa_lmem_bus_if (per_core_dxa_lmem_bus_if[c]),
            .dxa_txbar_bus_if(dxa_txbar_bus_if),
        `endif
            .empty           (mem_unit_empty[c]),
            .lsu_mem_if      (lsu_mem_if),
            .dcr_flush_if    (dcr_flush_dcache_if),
            .dcache_bus_if   (dcache_bus_if)
        );

        for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_dcache_passthru
            `ASSIGN_VX_MEM_BUS_IF (all_dc_core_if[c * DCACHE_NUM_REQS + i], dcache_bus_if[i]);
        end
    end

    // ========================================================================
    // Socket-shared DXA engine + staging-return switch
    // ========================================================================
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
    ) dxa_gmem_bus_if[1]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) dxa_lmem_bus_if[1]();

    VX_dcr_bus_if dxa_dcr_bus_if();
    assign dxa_dcr_bus_if.req_valid = dcr_req_valid;
    assign dxa_dcr_bus_if.req_data  = dcr_req_data;
    assign dcr_rsp_valid            = dxa_dcr_bus_if.rsp_valid;
    `UNUSED_VAR (dxa_dcr_bus_if.rsp_data)

    for (genvar c = 0; c < NC; ++c) begin : g_dxa_req_bind
        assign per_core_dxa_req_bus_if[c].req_valid = dxa_req_valid[c];
        assign per_core_dxa_req_bus_if[c].req_data  = dxa_req_data[c];
        assign dxa_req_ready[c] = per_core_dxa_req_bus_if[c].req_ready;
    end

    VX_dxa_core #(
        .INSTANCE_ID    (`SFORMATF(("%s-dxa", INSTANCE_ID))),
        .NUM_REQS       (NC),
        .GMEM_OUT_PORTS (1)
    ) dxa_core (
        .clk         (clk),
        .reset       (reset),
        .dcr_bus_if  (dxa_dcr_bus_if),
        .req_bus_if  (per_core_dxa_req_bus_if),
        .smem_bus_if (dxa_lmem_bus_if),
        .gmem_bus_if (dxa_gmem_bus_if),
        .busy        (dxa_busy)
    );

    // Route DXA lmem requests to per-core buses using core_local_id from tag.
    localparam DXA_LMEM_CORE_LOCAL_BITS = `CLOG2(NC);

    wire [`UP(DXA_LMEM_CORE_LOCAL_BITS)-1:0] dxa_lmem_core_sel;
    if (NC > 1) begin : g_dxa_lmem_sel
        assign dxa_lmem_core_sel = dxa_lmem_bus_if[0].req_data.tag.value[1 +: DXA_LMEM_CORE_LOCAL_BITS];
    end else begin : g_dxa_lmem_sel
        assign dxa_lmem_core_sel = '0;
    end

    VX_mem_bus_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NC),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W),
        .REQ_OUT_BUF ((NC > 1) ? 3 : 0),
        .RSP_OUT_BUF ((NC > 1) ? 3 : 0)
    ) dxa_lmem_core_switch (
        .clk        (clk),
        .reset      (reset),
        .bus_sel    (dxa_lmem_core_sel),
        .bus_in_if  (dxa_lmem_bus_if),
        .bus_out_if (per_core_dxa_lmem_bus_if)
    );

    // ========================================================================
    // Shared L1 dcache
    // ========================================================================
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_SECTOR_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if [L1_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-dcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_DCACHES),
        .NUM_INPUTS     (NC),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_DCACHE_SIZE),
        .LINE_SIZE      (DCACHE_LINE_SIZE),
        .SECTOR_SIZE    (DCACHE_SECTOR_SIZE),
        .NUM_BANKS      (DCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_DCACHE_NUM_WAYS),
        .WORD_SIZE      (DCACHE_WORD_SIZE),
        .NUM_REQS       (DCACHE_NUM_REQS),
        .MEM_PORTS      (L1_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_DCACHE_MREQ_SIZE),
        .LATENCY        (`VX_CFG_DCACHE_LATENCY),
        .TAG_WIDTH      (DCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_DCACHE_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_DCACHE_DIRTYBYTES),
        .REPL_POLICY    (`VX_CFG_DCACHE_REPL_POLICY),
        .NC_ENABLE      (1),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .IS_LLC         (DCACHE_IS_LLC),
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) dcache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (all_dc_core_if),
        .mem_bus_if     (dcache_mem_bus_if)
    );

    // ========================================================================
    // Socket memory arbiter: {icache (tied), dcache, DXA gmem} -> L2
    // ========================================================================
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (SOCKET_MEM_ARB_TAG_WIDTH)
    ) socket_mem_bus_if [L1_MEM_PORTS] ();

    for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin : g_mem_bus_if
        if (i == 0) begin : g_i0
            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
            ) socket_arb_in_if[SOCKET_MEM_ARB_REQS]();

            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_ARB_TAG_WIDTH)
            ) socket_arb_out_if[1]();

            // The icache slot is kept (production SOCKET_MEM_ARB_REQS shape,
            // "P" priority order preserved) with its request stream idle.
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_valid = 1'b0;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data  = '0;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_ready = 1'b0;
            `UNUSED_VAR (socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_ready)
            `UNUSED_VAR (socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_valid)
            `UNUSED_VAR (socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_data)

            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[DCACHE_MEM_ARB_IDX], dcache_mem_bus_if[0], SOCKET_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF (socket_arb_in_if[DXA_MEM_ARB_IDX], dxa_gmem_bus_if[0]);

            VX_mem_bus_arb #(
                .NUM_INPUTS (SOCKET_MEM_ARB_REQS),
                .NUM_OUTPUTS(1),
                .DATA_SIZE  (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH  (SOCKET_MEM_TAG_WIDTH),
                .TAG_SEL_IDX(0),
                .ARBITER    ("P"), // icache first, DXA last
                .REQ_OUT_BUF(3),
                .RSP_OUT_BUF(3)
            ) mem_arb (
                .clk        (clk),
                .reset      (reset),
                .bus_in_if  (socket_arb_in_if),
                .bus_out_if (socket_arb_out_if)
            );

            `ASSIGN_VX_MEM_BUS_IF (socket_mem_bus_if[0], socket_arb_out_if[0]);
        end else begin : g_i
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_mem_bus_if[i], dcache_mem_bus_if[i], SOCKET_MEM_ARB_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        end
    end

    // ========================================================================
    // Shared L2 (PASSTHRU wires when VX_CFG_L2_ENABLE is off)
    // ========================================================================
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) l2_core_if [L2_NUM_REQS] ();

    for (genvar i = 0; i < L2_NUM_REQS; ++i) begin : g_l2_clients
        `ASSIGN_VX_MEM_BUS_IF (l2_core_if[i], socket_mem_bus_if[i]);
    end

    VX_mem_bus_if #(
        .DATA_SIZE (L2_SECTOR_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) l2_mem_if [L2_MEM_PORTS] ();

    for (genvar i = 0; i < L2_MEM_PORTS; ++i) begin : g_l2_mem_bind
        assign l2_mem_req_valid[i]        = l2_mem_if[i].req_valid;
        assign l2_mem_req_rw[i]           = l2_mem_if[i].req_data.rw;
        assign l2_mem_req_byteen[i]       = l2_mem_if[i].req_data.byteen;
        assign l2_mem_req_addr[i]         = l2_mem_if[i].req_data.addr;
        assign l2_mem_req_tag[i]          = l2_mem_if[i].req_data.tag;
        assign l2_mem_req_data[i]         = l2_mem_if[i].req_data.data;
        assign l2_mem_if[i].req_ready     = l2_mem_req_ready[i];
        assign l2_mem_if[i].rsp_valid     = l2_mem_rsp_valid[i];
        assign l2_mem_if[i].rsp_data.data = l2_mem_rsp_data[i];
        assign l2_mem_if[i].rsp_data.tag  = l2_mem_rsp_tag[i];
        assign l2_mem_rsp_ready[i]        = l2_mem_if[i].rsp_ready;
        `UNUSED_VAR (l2_mem_if[i].req_data.attr)
    end

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`VX_CFG_L2_SIZE),
        .LINE_SIZE      (`VX_CFG_L2_LINE_SIZE),
        .SECTOR_SIZE    (L2_SECTOR_SIZE),
        .NUM_BANKS      (L2_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (L2_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_L2_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_L2_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_L2_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_L2_MREQ_SIZE),
        .LATENCY        (`VX_CFG_L2_LATENCY),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_L2_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_L2_DIRTYBYTES),
        .REPL_POLICY    (`VX_CFG_L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`VX_CFG_L2_ENABLED),
        .IS_LLC         (L2_IS_LLC),
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) l2cache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (l2_core_if),
        .mem_bus_if     (l2_mem_if)
    );

    assign empty = (& mem_unit_empty) && (& lsu_sched_empty) && !dxa_busy;

endmodule
