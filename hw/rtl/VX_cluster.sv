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

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // The raster engines are a launch master alongside the device KMU: a fragment
    // wave IS a launch now, so it merges here instead of being injected at every
    // core behind a private data bus.
    VX_kmu_bus_if raster_kmu_bus_if[1]();
    VX_kmu_bus_if kmu_arb_in_if[2]();

    assign kmu_arb_in_if[0].valid = kmu_bus_if[0].valid;
    assign kmu_arb_in_if[0].data  = kmu_bus_if[0].data;
    assign kmu_arb_in_if[0].kind  = kmu_bus_if[0].kind;
    assign kmu_arb_in_if[0].eop   = kmu_bus_if[0].eop;
    assign kmu_arb_in_if[0].dest  = kmu_bus_if[0].dest;
    assign kmu_bus_if[0].ready    = kmu_arb_in_if[0].ready;

    assign kmu_arb_in_if[1].valid = raster_kmu_bus_if[0].valid;
    assign kmu_arb_in_if[1].data  = raster_kmu_bus_if[0].data;
    assign kmu_arb_in_if[1].kind  = raster_kmu_bus_if[0].kind;
    assign kmu_arb_in_if[1].eop   = raster_kmu_bus_if[0].eop;
    assign kmu_arb_in_if[1].dest  = raster_kmu_bus_if[0].dest;
    assign raster_kmu_bus_if[0].ready = kmu_arb_in_if[1].ready;

    VX_kmu_bus_arb #(
        .NUM_INPUTS (2),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .DEST_LSB   (KMU_DEST_LSB_CLUSTER),
        .ARBITER    ("R"),
        // Two masters, so this arb always arbitrates -- even at NUM_SOCKETS == 1
        // it is never a 1:1 passthrough, hence both knobs are unconditional.
        .IN_BUF     (3),
        .OUT_BUF    (3)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_arb_in_if),
        .bus_out_if (per_socket_kmu_bus_if)
    );
`else
    VX_kmu_bus_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .DEST_LSB   (KMU_DEST_LSB_CLUSTER),
        .IN_BUF     ((NUM_SOCKETS > 1) ? 3 : 0),
        .OUT_BUF    ((NUM_SOCKETS > 1) ? 3 : 0)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_bus_if),
        .bus_out_if (per_socket_kmu_bus_if)
    );
`endif

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

    // L2 input buses (socket ports + cluster-resident gfx cache ports)
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) per_socket_mem_bus_if[L2_NUM_REQS]();

    // Socket memory output buses (socket arb output tag width)
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) socket_mem_bus_if[L2_SOCKET_REQS]();

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) rcache_l2_bus_if();
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) ocache_l2_bus_if();
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

    // Cluster DCR distribution — declared before its consumers (sockets/gfx).
`ifdef EXT_GFX_CLUSTER_ENABLE
    localparam NUM_DCR_GFX = 1;
    localparam DCR_GFX_IDX = NUM_SOCKETS;
`else
    localparam NUM_DCR_GFX = 0;
`endif
    localparam NUM_DCR_REQS = NUM_SOCKETS + NUM_DCR_GFX;
    VX_dcr_bus_if per_socket_dcr_bus_if[NUM_DCR_REQS]();
    VX_dcr_arb #(
        .NUM_REQS    (NUM_DCR_REQS),
        .REQ_OUT_BUF ((NUM_DCR_REQS > 1) ? 1 : 0),
        .RSP_OUT_BUF ((NUM_DCR_REQS > 1) ? 1 : 0)
    ) dcr_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_bus_if),
        .bus_out_if (per_socket_dcr_bus_if)
    );

`ifdef VX_CFG_EXT_OM_ENABLE
    // The socket->L2 trunk runs through VX_graphics, which peels the OM aperture
    // writes off it. The whole OM path -- steer, ingress, arb, cores -- lives there.
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) gfx_l2_out_bus_if [L2_SOCKET_REQS]();

    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[i], gfx_l2_out_bus_if[i]);
    end
`else
    // Sockets connect straight to their L2 input ports (TEX/RTU/DXA traffic
    // already merged at each socket's L2-facing arb).
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[i], socket_mem_bus_if[i]);
    end
`endif

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

            .mem_bus_if     (socket_mem_bus_if[socket_id * L1_MEM_PORTS +: L1_MEM_PORTS]),

        `ifdef VX_CFG_EXT_OM_ENABLE
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

`ifdef EXT_GFX_CLUSTER_ENABLE
    // Producer busy from the graphics block (raster engine out-of-band drain).
    wire gfx_busy;

    // Alias the graphics block's dedicated DCR array element onto a scalar
    // interface: a constant array index in a modport binding is rejected by
    // sv2v. Pure net joins, zero cost.
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
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_perf              (gfx_raster_perf),
        .rcache_perf              (gfx_rcache_perf),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .om_perf                  (gfx_om_perf),
        .ocache_perf              (gfx_ocache_perf),
    `endif
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_kmu_bus_if (raster_kmu_bus_if),
        .rcache_mem_bus_if        (rcache_l2_bus_if),
        .raster_launch_if         (raster_launch_if),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .socket_mem_bus_if        (socket_mem_bus_if),
        .l2_out_bus_if            (gfx_l2_out_bus_if),
        .ocache_mem_bus_if        (ocache_l2_bus_if),
    `endif
        .dcr_bus_if               (gfx_dcr_bus_if),
        .cluster_flush_if         (cluster_flush_if),
        .busy                     (gfx_busy)
    );

`ifdef VX_CFG_EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_RASTER_IDX], rcache_l2_bus_if);
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[L2_GFX_OM_IDX], ocache_l2_bus_if);
`endif

`elsif EXT_GFX_ANY_ENABLE
    // No cluster-resident graphics caches: the socket-local flush legs
    // (tcache/rtcache) complete on their own; this level contributes done=1.
    assign cluster_flush_if.done = 1'b1;
    `UNUSED_VAR (cluster_flush_if.req)
    wire gfx_busy = 1'b0;
`endif // EXT_GFX_CLUSTER_ENABLE

    wire busy_r;
`ifdef EXT_GFX_ANY_ENABLE
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_socket_busy) | gfx_busy, 1'b1, 1, (NUM_SOCKETS > 1));
`else
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));
`endif
    assign busy = busy_r | dcr_bus_if.req_valid;

endmodule
