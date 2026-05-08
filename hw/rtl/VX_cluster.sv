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

module VX_cluster import VX_gpu_pkg::*;
`ifdef EXT_DXA_ENABLE
    import VX_dxa_pkg::*;
`endif
#(
    parameter CLUSTER_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave         dcr_bus_if,

    // Memory
    VX_mem_bus_if.master        mem_bus_if [`L2_MEM_PORTS],

    // KMU bus
    VX_kmu_bus_if.slave         kmu_bus_if[1],

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (NUM_SOCKETS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l2_perf;
    sysmem_perf_t sysmem_perf_tmp;
`ifdef EXT_DXA_ENABLE
    dxa_perf_t dxa_core_perf;
`endif
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.l2cache = l2_perf;
    `ifdef EXT_DXA_ENABLE
        sysmem_perf_tmp.dxa = dxa_core_perf;
    `endif
    end
`endif

    VX_kmu_bus_if per_socket_kmu_bus_if[NUM_SOCKETS]();

    VX_kmu_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (NUM_SOCKETS)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_bus_if),
        .bus_out_if (per_socket_kmu_bus_if)
    );

    VX_gbar_bus_if per_socket_gbar_bus_if[NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    VX_gbar_arb #(
        .NUM_REQS (NUM_SOCKETS),
        .REQ_OUT_BUF ((NUM_SOCKETS > 2) ? 2 : 0),
        .RSP_OUT_BUF ((NUM_SOCKETS > 2) ? 2 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID (`SFORMATF(("gbar%0d", CLUSTER_ID)))
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if)
    );

    // L2 input buses (post-arb tag width when DXA enabled)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) per_socket_mem_bus_if[L2_NUM_REQS]();

    // Socket L1 output buses (pre-arb, original tag width)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) socket_mem_bus_if[L2_SOCKET_REQS]();

`ifdef EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB1_TAG_WIDTH)
    ) per_socket_tex_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) tcache_l2_bus_if();
`endif

`ifdef EXT_RASTER_ENABLE
    VX_raster_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) per_socket_raster_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) rcache_l2_bus_if();
`endif

`ifdef EXT_OM_ENABLE
    VX_om_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) per_socket_om_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) ocache_l2_bus_if();
`endif

`ifdef EXT_DXA_ENABLE
    import VX_dxa_pkg::*;
    VX_dxa_req_bus_if per_socket_dxa_req_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) dxa_gmem_bus_if[DXA_L2_GMEM_PORTS]();
    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) dxa_lmem_bus_if[1]();
    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) per_socket_dxa_lmem_bus_if[NUM_SOCKETS]();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) l2_mem_bus_if[`L2_MEM_PORTS]();

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (`L2_MEM_PORTS),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (L2_MREQ_SIZE),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`L2_WRITEBACK),
        .DIRTY_BYTES    (`L2_DIRTYBYTES),
        .REPL_POLICY    (`L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED),
        .IS_LLC         (L2_IS_LLC),
        .AMO_ENABLE     (`EXT_A_ENABLED && L2_IS_LLC)
    ) l2cache (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .cache_perf     (l2_perf),
    `endif
        .core_bus_if    (per_socket_mem_bus_if),
        .mem_bus_if     (l2_mem_bus_if)
    );

`ifdef EXT_DXA_ENABLE
    VX_dxa_core #(
        .INSTANCE_ID      (`SFORMATF(("%s-dxa-core", INSTANCE_ID))),
        .NUM_REQS     (NUM_SOCKETS),
        .NUM_DXA_UNITS    (`NUM_DXA_UNITS),
        .GMEM_OUT_PORTS   (DXA_L2_GMEM_PORTS)
    ) dxa_core (
        .clk              (clk),
        .reset            (reset),
    `ifdef PERF_ENABLE
        .dxa_perf         (dxa_core_perf),
    `endif
        .dcr_bus_if       (per_socket_dcr_bus_if[NUM_SOCKETS]),
        .req_bus_if       (per_socket_dxa_req_bus_if),
        .smem_bus_if      (dxa_lmem_bus_if),
        .gmem_bus_if      (dxa_gmem_bus_if),
        `UNUSED_PIN (busy)
    );

    // Route DXA lmem requests to per-socket buses using core_id from tag.
    // Tag value layout: {core_id[NC_BITS-1:0], engine_value[0]}
    // socket_id = core_id[CORE_LOCAL_BITS +: SOCKET_SEL_BITS]
    localparam DXA_LMEM_CORE_LOCAL_BITS = `CLOG2(`SOCKET_SIZE);
    localparam DXA_LMEM_SOCKET_SEL_BITS = `CLOG2(NUM_SOCKETS);
    wire [`UP(DXA_LMEM_SOCKET_SEL_BITS)-1:0] dxa_lmem_socket_sel;
    if (NUM_SOCKETS > 1) begin : g_dxa_lmem_sel
        assign dxa_lmem_socket_sel = dxa_lmem_bus_if[0].req_data.tag.value[1 + DXA_LMEM_CORE_LOCAL_BITS +: DXA_LMEM_SOCKET_SEL_BITS];
    end else begin : g_dxa_lmem_sel
        assign dxa_lmem_socket_sel = '0;
    end

    VX_mem_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) dxa_lmem_socket_switch (
        .clk        (clk),
        .reset      (reset),
        .bus_sel    (dxa_lmem_socket_sel),
        .bus_in_if  (dxa_lmem_bus_if),
        .bus_out_if (per_socket_dxa_lmem_bus_if)
    );

    // LSU+DXA arb: LSU gets priority ("P") to prevent DXA bulk traffic from
    // starving core icache/dcache at L2 and the shared memory bus.
    // Lower index = higher priority, so LSU is bound first.
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) l2_arb_in_if[2 * L2_SOCKET_REQS]();

    // Bind LSU ports first (high priority, indices 0..L2_SOCKET_REQS-1)
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_lsu_l2_bind
        `ASSIGN_VX_MEM_BUS_IF (l2_arb_in_if[i], socket_mem_bus_if[i]);
    end

    // Bind DXA gmem ports second (low priority, indices L2_SOCKET_REQS+..)
    for (genvar i = 0; i < DXA_L2_GMEM_PORTS; ++i) begin : g_dxa_l2_bind
        `ASSIGN_VX_MEM_BUS_IF (l2_arb_in_if[L2_SOCKET_REQS + i], dxa_gmem_bus_if[i]);
    end

    // Tie off unused DXA slots
    for (genvar i = DXA_L2_GMEM_PORTS; i < L2_SOCKET_REQS; ++i) begin : g_dxa_l2_tieoff
        assign l2_arb_in_if[L2_SOCKET_REQS + i].req_valid = 1'b0;
        assign l2_arb_in_if[L2_SOCKET_REQS + i].req_data  = '0;
        assign l2_arb_in_if[L2_SOCKET_REQS + i].rsp_ready = 1'b1;
    end

    VX_mem_arb #(
        .NUM_INPUTS  (2 * L2_SOCKET_REQS),
        .NUM_OUTPUTS (L2_SOCKET_REQS),
        .DATA_SIZE   (`L1_LINE_SIZE),
        .TAG_WIDTH   (L1_MEM_ARB_TAG_WIDTH),
        .TAG_SEL_IDX (0),
        .ARBITER     ("P"),
        // DXA fix: add 1-entry elastic buffer per response output.
        // After DXA transfers complete, the last DXA-tagged response leaves a
        // stale routing bit in the L2 bank's tag register. VX_stream_switch
        // propagates this stale sel_in to ready_in, permanently backpressuring
        // the L2 bank even when no response is pending. Adding RSP_OUT_BUF=1
        // ensures the DXA port's buffer is always empty (and ready=1) after DXA
        // completes, breaking the spurious backpressure chain.
        .RSP_OUT_BUF (1)
    ) dxa_l2_priority_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (l2_arb_in_if),
        .bus_out_if (per_socket_mem_bus_if)
    );

`else
    // No DXA: direct socket → L2
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_no_dxa_l2
        `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[i], socket_mem_bus_if[i]);
    end
`endif

    for (genvar i = 0; i < `L2_MEM_PORTS; ++i) begin : g_l2_mem_out
        `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], l2_mem_bus_if[i]);
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_SOCKETS-1:0] per_socket_busy;

`ifdef EXT_GFX_ANY_ENABLE
    localparam NUM_DCR_GFX = 1;
    localparam DCR_GFX_IDX = NUM_SOCKETS + `EXT_DXA_ENABLED;
`else
    localparam NUM_DCR_GFX = 0;
`endif
    localparam NUM_DCR_REQS = NUM_SOCKETS + `EXT_DXA_ENABLED + NUM_DCR_GFX;
    VX_dcr_bus_if per_socket_dcr_bus_if[NUM_DCR_REQS]();
    VX_dcr_arb #(
        .NUM_REQS    (NUM_DCR_REQS),
        .REQ_OUT_BUF ((NUM_DCR_REQS > 1) ? 1 : 0)
    ) dcr_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_bus_if),
        .bus_out_if (per_socket_dcr_bus_if)
    );

    // Generate all sockets
    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, socket_id)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+socket_id)

            .clk            (clk),
            .reset          (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (per_socket_dcr_bus_if[socket_id]),

            .mem_bus_if     (socket_mem_bus_if[socket_id * `L1_MEM_PORTS +: `L1_MEM_PORTS]),

        `ifdef EXT_DXA_ENABLE
            .dxa_req_bus_if (per_socket_dxa_req_bus_if[socket_id]),
            .dxa_lmem_bus_if(per_socket_dxa_lmem_bus_if[socket_id +: 1]),
        `endif

        `ifdef EXT_TEX_ENABLE
            .per_socket_tex_bus_if (per_socket_tex_bus_if[socket_id]),
        `endif

        `ifdef EXT_OM_ENABLE
            .per_socket_om_bus_if (per_socket_om_bus_if[socket_id]),
        `endif

        `ifdef EXT_RASTER_ENABLE
            .per_socket_raster_bus_if (per_socket_raster_bus_if[socket_id]),
        `endif

            .kmu_bus_if     (per_socket_kmu_bus_if[socket_id +: 1]),

            .gbar_bus_if    (per_socket_gbar_bus_if[socket_id]),

            .busy           (per_socket_busy[socket_id])
        );
    end

    ///////////////////////////////////////////////////////////////////////////
    // Graphics extensions cluster integration
    ///////////////////////////////////////////////////////////////////////////

`ifdef EXT_GFX_ANY_ENABLE
    VX_graphics #(
        .CLUSTER_ID (CLUSTER_ID)
    ) graphics (
        .clk        (clk),
        .reset      (reset),
    `ifdef EXT_TEX_ENABLE
        .per_socket_tex_bus_if    (per_socket_tex_bus_if),
        .tcache_mem_bus_if        (tcache_l2_bus_if),
    `endif
    `ifdef EXT_RASTER_ENABLE
        .per_socket_raster_bus_if (per_socket_raster_bus_if),
        .rcache_mem_bus_if        (rcache_l2_bus_if),
    `endif
    `ifdef EXT_OM_ENABLE
        .per_socket_om_bus_if     (per_socket_om_bus_if),
        .ocache_mem_bus_if        (ocache_l2_bus_if),
    `endif
        .dcr_bus_if               (per_socket_dcr_bus_if[DCR_GFX_IDX])
    );

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_TEX_IDX], tcache_l2_bus_if);
`endif

`ifdef EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_RASTER_IDX], rcache_l2_bus_if);
`endif

`ifdef EXT_OM_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_OM_IDX], ocache_l2_bus_if);
`endif

`endif // EXT_GFX_ANY_ENABLE

    wire busy_r;
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));
    assign busy = busy_r | dcr_bus_if.req_valid;

endmodule
