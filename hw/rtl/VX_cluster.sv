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
`ifdef VX_CFG_EXT_DXA_ENABLE
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
    VX_mem_bus_if.master        mem_bus_if [L2_MEM_PORTS],

    // KMU bus
    VX_kmu_bus_if.slave         kmu_bus_if[1],

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // Delegated draw launch (device KMU → raster engines)
    VX_raster_launch_if.slave   raster_launch_if[1],
`endif

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
`ifdef VX_CFG_EXT_DXA_ENABLE
    dxa_perf_t per_socket_dxa_perf[NUM_SOCKETS];
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    tex_perf_t   gfx_tex_perf;
    cache_perf_t gfx_tcache_perf;
`endif
`ifdef VX_CFG_EXT_RASTER_ENABLE
    raster_perf_t gfx_raster_perf;
    cache_perf_t  gfx_rcache_perf;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    om_perf_t    gfx_om_perf;
    cache_perf_t gfx_ocache_perf;
`endif
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.l2cache = l2_perf;
    `ifdef VX_CFG_EXT_TEX_ENABLE
        sysmem_perf_tmp.tex    = gfx_tex_perf;
        sysmem_perf_tmp.tcache = gfx_tcache_perf;
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        sysmem_perf_tmp.raster = gfx_raster_perf;
        sysmem_perf_tmp.rcache = gfx_rcache_perf;
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        sysmem_perf_tmp.om     = gfx_om_perf;
        sysmem_perf_tmp.ocache = gfx_ocache_perf;
    `endif
    end
`endif

    VX_kmu_bus_if per_socket_kmu_bus_if[NUM_SOCKETS]();

    VX_kmu_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .OUT_BUF    ((NUM_SOCKETS > 1) ? 3 : 0)
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
        .REQ_OUT_BUF ((NUM_SOCKETS > 1) ? 3 : 0),
        .RSP_OUT_BUF ((NUM_SOCKETS > 1) ? 3 : 0)
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
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) per_socket_mem_bus_if[L2_NUM_REQS]();

    // Socket outputs already include the socket-local L1-versus-DXA arb tag.
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) socket_mem_bus_if[L2_SOCKET_REQS]();

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB1_TAG_WIDTH)
    ) per_socket_tex_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) tcache_l2_bus_if();
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES)
    ) per_socket_raster_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) rcache_l2_bus_if();
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_om_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES)
    ) per_socket_om_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) ocache_l2_bus_if();
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (RTU_REQ_ARB1_TAG_WIDTH)
    ) per_socket_rtu_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) rtcache_l2_bus_if();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (L2_SECTOR_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) l2_mem_bus_if[L2_MEM_PORTS]();

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
    `ifdef PERF_ENABLE
        .cache_perf     (l2_perf),
    `endif
        .core_bus_if    (per_socket_mem_bus_if),
        .mem_bus_if     (l2_mem_bus_if)
    );

    // Cluster DCR distribution. Each socket performs its own local fan-out to
    // cores and the socket-owned DXA endpoint.
`ifdef EXT_GFX_ANY_ENABLE
    localparam NUM_DCR_GFX = 1;
    localparam DCR_GFX_IDX = NUM_SOCKETS;
`else
    localparam NUM_DCR_GFX = 0;
`endif
    localparam NUM_DCR_REQS = NUM_SOCKETS + NUM_DCR_GFX;
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

    // Socket traffic, including socket-local DXA GMEM traffic, reaches L2 as
    // one already-arbitrated stream per socket memory port.
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[i], socket_mem_bus_if[i]);
    end

    for (genvar i = 0; i < L2_MEM_PORTS; ++i) begin : g_l2_mem_out
        `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], l2_mem_bus_if[i]);
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_SOCKETS-1:0] per_socket_busy;

`ifdef EXT_GFX_ANY_ENABLE
    // OR all socket flush reqs together; broadcast .done to every socket.
    // Sockets fire their reqs sequentially; the shared resource flushes
    // once per req (subsequent reqs see an already-empty cache).
    VX_dcr_flush_if cluster_flush_if();
    VX_dcr_flush_if per_socket_cluster_flush_if [NUM_SOCKETS]();
    wire [NUM_SOCKETS-1:0] per_socket_cluster_flush_req;
    for (genvar s = 0; s < NUM_SOCKETS; ++s) begin : g_per_socket_cluster_flush
        assign per_socket_cluster_flush_req[s]    = per_socket_cluster_flush_if[s].req;
        assign per_socket_cluster_flush_if[s].done = cluster_flush_if.done;
    end
    assign cluster_flush_if.req = (| per_socket_cluster_flush_req);
`endif

    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

    `ifdef PERF_ENABLE
        // DXA counters are socket-owned.  Feed each core only its socket's
        // counters so runtime aggregation over one representative core per
        // socket does not count the cluster total once for every socket.
        sysmem_perf_t socket_sysmem_perf;
        always @(*) begin
            socket_sysmem_perf = sysmem_perf_tmp;
        `ifdef VX_CFG_EXT_DXA_ENABLE
            socket_sysmem_perf.dxa = per_socket_dxa_perf[socket_id];
        `endif
        end
    `endif

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, socket_id)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+socket_id)

            .clk            (clk),
            .reset          (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (socket_sysmem_perf),
        `ifdef VX_CFG_EXT_DXA_ENABLE
            .dxa_perf       (per_socket_dxa_perf[socket_id]),
        `endif
        `endif

            .dcr_bus_if     (per_socket_dcr_bus_if[socket_id]),

            .mem_bus_if     (socket_mem_bus_if[socket_id * L1_MEM_PORTS +: L1_MEM_PORTS]),

        `ifdef VX_CFG_EXT_TEX_ENABLE
            .per_socket_tex_bus_if (per_socket_tex_bus_if[socket_id]),
        `endif

        `ifdef VX_CFG_EXT_OM_ENABLE
            .per_socket_om_bus_if (per_socket_om_bus_if[socket_id]),
        `endif

        `ifdef VX_CFG_EXT_RASTER_ENABLE
            .per_socket_raster_bus_if (per_socket_raster_bus_if[socket_id]),
        `endif

        `ifdef VX_CFG_EXT_RTU_ENABLE
            .per_socket_rtu_bus_if (per_socket_rtu_bus_if[socket_id]),
        `endif

        `ifdef EXT_GFX_ANY_ENABLE
            .cluster_flush_if (per_socket_cluster_flush_if[socket_id]),
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
    // Producer busy from the graphics block (raster engine out-of-band drain).
    wire gfx_busy;

    // Alias the graphics block's dedicated DCR array element onto a scalar
    // interface (same rationale as the DXA binding above): a constant array
    // index in a modport binding is rejected by sv2v. Pure net joins, zero cost.
    VX_dcr_bus_if gfx_dcr_bus_if();
    assign gfx_dcr_bus_if.req_valid                     = per_socket_dcr_bus_if[DCR_GFX_IDX].req_valid;
    assign gfx_dcr_bus_if.req_data                      = per_socket_dcr_bus_if[DCR_GFX_IDX].req_data;
    assign per_socket_dcr_bus_if[DCR_GFX_IDX].rsp_valid = gfx_dcr_bus_if.rsp_valid;
    assign per_socket_dcr_bus_if[DCR_GFX_IDX].rsp_data  = gfx_dcr_bus_if.rsp_data;

    VX_graphics #(
        .CLUSTER_ID (CLUSTER_ID)
    ) graphics (
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_perf                 (gfx_tex_perf),
        .tcache_perf              (gfx_tcache_perf),
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_perf              (gfx_raster_perf),
        .rcache_perf              (gfx_rcache_perf),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .om_perf                  (gfx_om_perf),
        .ocache_perf              (gfx_ocache_perf),
    `endif
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .per_socket_tex_bus_if    (per_socket_tex_bus_if),
        .tcache_mem_bus_if        (tcache_l2_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .per_socket_raster_bus_if (per_socket_raster_bus_if),
        .rcache_mem_bus_if        (rcache_l2_bus_if),
        .raster_launch_if         (raster_launch_if),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .per_socket_om_bus_if     (per_socket_om_bus_if),
        .ocache_mem_bus_if        (ocache_l2_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .per_socket_rtu_bus_if    (per_socket_rtu_bus_if),
        .rtcache_mem_bus_if       (rtcache_l2_bus_if),
    `endif
        .dcr_bus_if               (gfx_dcr_bus_if),
        .cluster_flush_if         (cluster_flush_if),
        .busy                     (gfx_busy)
    );

`ifdef VX_CFG_EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_TEX_IDX], tcache_l2_bus_if);
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_RASTER_IDX], rcache_l2_bus_if);
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_OM_IDX], ocache_l2_bus_if);
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_RTU_IDX], rtcache_l2_bus_if);
`endif

`endif // EXT_GFX_ANY_ENABLE

    wire busy_r;
`ifdef EXT_GFX_ANY_ENABLE
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_socket_busy) | gfx_busy, 1'b1, 1, (NUM_SOCKETS > 1));
`else
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));
`endif
    assign busy = busy_r | dcr_bus_if.req_valid;

endmodule
