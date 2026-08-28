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

// Standalone graphics-subsystem synthesis sandbox.
//
// Reproduces one cluster's graphics slice (one socket + the cluster-resident
// units) exactly as VX_socket/VX_cluster wire it, with the cores removed:
//
//   socket slice:
//     VX_tex_bus_arb  -> VX_tex_core xN -> tcache (VX_cache_cluster)
//     VX_rtu_bus_arb  -> VX_rtu_core xN -> rtcache (VX_cache_cluster)
//     VX_mem_bus_arb  : the socket memory arbiter with its full production
//                       input set {icache, dcache, tcache, rtcache, DXA gmem};
//                       the icache/dcache/DXA slots are driven from top ports
//   cluster tail:
//     VX_graphics     : the production container — raster launch fork,
//                       VX_raster_core xN + rcache, the OM trunk stage
//                       (VX_om_steer/VX_om_ingress per trunk port),
//                       VX_om_bus_arb -> VX_om_core xN + ocache
//     VX_cache_wrap L2: real client composition (socket trunk +
//                       dedicated rcache/ocache ports); PASSTHRU when
//                       VX_CFG_L2_ENABLE is off
//
// OM fragment exports are injected at the dcache arbiter slot as ordinary
// stores tagged with the OM-aperture attribute — the exact traffic shape the
// om_steer sees in production (fragment exports ride the LSU/dcache path).
//
// Fidelity: every engine<->cache, cache<->arbiter, trunk<->steer and
// client<->L2 boundary is present and loaded by its real neighbour. What is
// removed (cores: decode/issue/SFU front-ends, LSU, dcache internals) never
// drove one of those nets — the TEX/RTU buses, the draw-kick token and the
// trunk streams are the complete core-facing surface.
//
// No component may be optimized out: every DUT input is driven by a
// top-level port and every DUT output is observed at a top-level port.

module VX_gfx_top import VX_gpu_pkg::*
`ifdef VX_CFG_EXT_TEX_ENABLE
    , VX_tex_pkg::*
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    , VX_rtu_pkg::*
`endif
; #(
    parameter `STRING INSTANCE_ID = "gfx_top",
    // Derived port geometry (do not override).
    parameter NC            = `VX_CFG_SOCKET_SIZE,
    parameter SFU_LANES     = `VX_CFG_NUM_SFU_LANES,
    parameter L2_MEM_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(L2_SECTOR_SIZE),
    parameter EXT_ADDR_W    = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_L1_LINE_SIZE)
) (
    input  wire clk,
    input  wire reset,

    // -----------------------------------------------------------------------
    // DCR write stream (tex/raster/om/rtu state; each unit filters by range)
    // -----------------------------------------------------------------------
    input  wire      dcr_req_valid,
    input  dcr_req_t dcr_req_data,
    output wire      dcr_rsp_valid,   // observed: keeps the DCR slaves live

`ifdef VX_CFG_EXT_TEX_ENABLE
    // -----------------------------------------------------------------------
    // TEX request/response streams, per core
    // -----------------------------------------------------------------------
    input  wire [NC-1:0]                                    tex_req_valid,
    input  wire [NC-1:0][SFU_LANES-1:0]                     tex_req_mask,
    input  wire [NC-1:0][1:0][SFU_LANES-1:0][31:0]          tex_req_coords,
    input  wire [NC-1:0][SFU_LANES-1:0][TEX_LOD_BITS-1:0]   tex_req_lod,
    input  wire [NC-1:0][TEX_STAGE_BITS-1:0]                tex_req_stage,
    input  wire [NC-1:0][TEX_REQ_TAG_WIDTH-1:0]             tex_req_tag,
    output wire [NC-1:0]                                    tex_req_ready,

    output wire [NC-1:0]                                    tex_rsp_valid,
    output wire [NC-1:0][SFU_LANES-1:0][31:0]               tex_rsp_texels,
    output wire [NC-1:0][TEX_REQ_TAG_WIDTH-1:0]             tex_rsp_tag,
    input  wire [NC-1:0]                                    tex_rsp_ready,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    // -----------------------------------------------------------------------
    // RTU arm/req/win channels, per core
    // -----------------------------------------------------------------------
    input  wire [NC-1:0]                                    rtu_arm_valid,
    input  wire [NC-1:0][NW_WIDTH-1:0]                      rtu_arm_wid,
    input  wire [NC-1:0][SFU_LANES-1:0]                     rtu_arm_mask,
    input  wire [NC-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0]        rtu_arm_scene_base,
    input  wire [NC-1:0][15:0]                              rtu_arm_flags,
    input  wire [NC-1:0][15:0]                              rtu_arm_cull_mask,
    input  wire [NC-1:0][31:0]                              rtu_arm_payload_ptr,
    input  wire [NC-1:0][RTU_REQ_TAG_WIDTH-1:0]             rtu_arm_tag,
    output wire [NC-1:0]                                    rtu_arm_ready,

    input  wire [NC-1:0]                                    rtu_req_valid,
    input  wire [NC-1:0]                                    rtu_req_kind,
    input  wire [NC-1:0][NW_WIDTH-1:0]                      rtu_req_wid,
    input  wire [NC-1:0][SFU_LANES-1:0][31:0]               rtu_req_data,
    input  wire [NC-1:0][SFU_LANES-1:0][RTU_CB_ACTION_BITS-1:0] rtu_req_cb_action,
    output wire [NC-1:0]                                    rtu_req_ready,

    output wire [NC-1:0]                                    rtu_win_valid,
    output wire [NC-1:0]                                    rtu_win_is_cand,
    output wire [NC-1:0][NW_WIDTH-1:0]                      rtu_win_wid,
    output wire [NC-1:0][RTU_SLOT_BITS-1:0]                 rtu_win_slot,
    output wire [NC-1:0][SFU_LANES-1:0]                     rtu_win_mask,
    output wire [NC-1:0][SFU_LANES-1:0][31:0]               rtu_win_data,
    output wire [NC-1:0][RTU_REQ_TAG_WIDTH-1:0]             rtu_win_tag,
    input  wire [NC-1:0]                                    rtu_win_ready,
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // -----------------------------------------------------------------------
    // Raster: draw kick in, merged fragment-launch stream out
    // -----------------------------------------------------------------------
    input  wire                     raster_launch_valid,
    output wire                     raster_launch_ready,

    output wire                     kmu_valid,
    output wire                     kmu_kind,
    output wire                     kmu_eop,
    output wire [KMU_DEST_W-1:0]    kmu_dest,
    output wire [KMU_DATAW-1:0]     kmu_data,
    input  wire                     kmu_ready,
`endif

    // -----------------------------------------------------------------------
    // Socket mem_arb external slots (icache / dcache / DXA gmem).
    // OM fragment exports are injected on the dcache slot with the
    // OM-aperture attribute set inside the request's physical address range.
    // -----------------------------------------------------------------------
    input  wire                                 ic_mem_req_valid,
    input  wire [EXT_ADDR_W-1:0]                ic_mem_req_addr,
    input  wire [SOCKET_MEM_TAG_WIDTH-1:0]      ic_mem_req_tag,
    output wire                                 ic_mem_req_ready,
    output wire                                 ic_mem_rsp_valid,
    output wire [`VX_CFG_L1_LINE_SIZE*8-1:0]    ic_mem_rsp_data,
    output wire [SOCKET_MEM_TAG_WIDTH-1:0]      ic_mem_rsp_tag,
    input  wire                                 ic_mem_rsp_ready,

    input  wire [L1_MEM_PORTS-1:0]                          dc_mem_req_valid,
    input  wire [L1_MEM_PORTS-1:0]                          dc_mem_req_rw,
    input  wire [L1_MEM_PORTS-1:0][`VX_CFG_L1_LINE_SIZE-1:0] dc_mem_req_byteen,
    input  wire [L1_MEM_PORTS-1:0][EXT_ADDR_W-1:0]          dc_mem_req_addr,
    input  wire [L1_MEM_PORTS-1:0][MEM_ATTR_WIDTH-1:0]      dc_mem_req_attr,
    input  wire [L1_MEM_PORTS-1:0][`VX_CFG_L1_LINE_SIZE*8-1:0] dc_mem_req_data,
    input  wire [L1_MEM_PORTS-1:0][SOCKET_MEM_TAG_WIDTH-1:0] dc_mem_req_tag,
    output wire [L1_MEM_PORTS-1:0]                          dc_mem_req_ready,
    output wire [L1_MEM_PORTS-1:0]                          dc_mem_rsp_valid,
    output wire [L1_MEM_PORTS-1:0][`VX_CFG_L1_LINE_SIZE*8-1:0] dc_mem_rsp_data,
    output wire [L1_MEM_PORTS-1:0][SOCKET_MEM_TAG_WIDTH-1:0] dc_mem_rsp_tag,
    input  wire [L1_MEM_PORTS-1:0]                          dc_mem_rsp_ready,

    input  wire                                 dxa_mem_req_valid,
    input  wire [EXT_ADDR_W-1:0]                dxa_mem_req_addr,
    input  wire [SOCKET_MEM_TAG_WIDTH-1:0]      dxa_mem_req_tag,
    output wire                                 dxa_mem_req_ready,
    output wire                                 dxa_mem_rsp_valid,
    output wire [`VX_CFG_L1_LINE_SIZE*8-1:0]    dxa_mem_rsp_data,
    output wire [SOCKET_MEM_TAG_WIDTH-1:0]      dxa_mem_rsp_tag,
    input  wire                                 dxa_mem_rsp_ready,

    // -----------------------------------------------------------------------
    // Cache flush handshake (tcache + rtcache + rcache + ocache)
    // -----------------------------------------------------------------------
    input  wire flush_req,
    output wire flush_done,

    // -----------------------------------------------------------------------
    // L2 memory side (master)
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

    output wire busy
);
    // One socket per cluster in this sandbox: the trunk width equals the
    // socket's memory-port count.
    `STATIC_ASSERT ((L2_SOCKET_REQS == L1_MEM_PORTS), ("gfx_top: single-socket sandbox"))

    wire [2:0] flush_done_w;

    ///////////////////////////////////////////////////////////////////////////
    // TEX — socket-resident texture units + private tcache
    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (SFU_LANES),
        .TAG_WIDTH (TEX_REQ_TAG_WIDTH)
    ) per_core_tex_bus_if[NC]();

    for (genvar c = 0; c < NC; ++c) begin : g_tex_bind
        assign per_core_tex_bus_if[c].req_valid       = tex_req_valid[c];
        assign per_core_tex_bus_if[c].req_data.mask   = tex_req_mask[c];
        assign per_core_tex_bus_if[c].req_data.coords = tex_req_coords[c];
        assign per_core_tex_bus_if[c].req_data.lod    = tex_req_lod[c];
        assign per_core_tex_bus_if[c].req_data.stage  = tex_req_stage[c];
        assign per_core_tex_bus_if[c].req_data.tag    = tex_req_tag[c];
        assign tex_req_ready[c]  = per_core_tex_bus_if[c].req_ready;
        assign tex_rsp_valid[c]  = per_core_tex_bus_if[c].rsp_valid;
        assign tex_rsp_texels[c] = per_core_tex_bus_if[c].rsp_data.texels;
        assign tex_rsp_tag[c]    = per_core_tex_bus_if[c].rsp_data.tag;
        assign per_core_tex_bus_if[c].rsp_ready = tex_rsp_ready[c];
    end

    VX_tex_bus_if #(
        .NUM_LANES (SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB1_TAG_WIDTH)
    ) tex_bus_if [`VX_CFG_NUM_TEX_CORES] ();

    VX_tex_bus_arb #(
        .NUM_INPUTS  (NC),
        .NUM_LANES   (SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_TEX_CORES),
        .TAG_WIDTH   (TEX_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_REQ ((NC != `VX_CFG_NUM_TEX_CORES) ? 3 : 0),
        .OUT_BUF_RSP ((NC != `VX_CFG_NUM_TEX_CORES) ? 3 : 0)
    ) tex_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_tex_bus_if),
        .bus_out_if (tex_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_bus_if [`VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS] ();

    for (genvar i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin : g_tex_core
        VX_dcr_bus_if tex_dcr_bus_if();
        assign tex_dcr_bus_if.req_valid = dcr_req_valid;
        assign tex_dcr_bus_if.req_data  = dcr_req_data;
        `UNUSED_VAR (tex_dcr_bus_if.rsp_valid)
        `UNUSED_VAR (tex_dcr_bus_if.rsp_data)

        VX_tex_core #(
            .INSTANCE_ID (`SFORMATF(("%s-tex%0d", INSTANCE_ID, i))),
            .NUM_LANES   (SFU_LANES),
            .TAG_WIDTH   (TEX_REQ_ARB1_TAG_WIDTH)
        ) tex_core (
            .clk          (clk),
            .reset        (reset),
            .dcr_bus_if   (tex_dcr_bus_if),
            .tex_bus_if   (tex_bus_if[i]),
            .cache_bus_if (tcache_bus_if[i * TCACHE_NUM_REQS +: TCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_tmp_if [TCACHE_MEM_PORTS] ();

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_BUS_TAG_WIDTH)
    ) tcache_flushable_bus_if [`VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS] ();

    VX_dcr_flush_if tcache_flush_if();
    assign tcache_flush_if.req = flush_req;
    assign flush_done_w[0] = tcache_flush_if.done;

    VX_dcr_flush #(
        .WORD_SIZE   (TCACHE_WORD_SIZE),
        .TAG_WIDTH   (TCACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3)
    ) tcache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (tcache_flush_if),
        .core_bus_if  (tcache_bus_if[0]),
        .cache_bus_if (tcache_flushable_bus_if[0])
    );

    for (genvar i = 1; i < `VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS; ++i) begin : g_tcache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (tcache_flushable_bus_if[i], tcache_bus_if[i],
                                  TCACHE_BUS_TAG_WIDTH, TCACHE_TAG_WIDTH, 0);
    end

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-tcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_TCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_TEX_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_TCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_TCACHE_NUM_WAYS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .MEM_PORTS      (TCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_BUS_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3)
    ) tcache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (tcache_flushable_bus_if),
        .mem_bus_if     (tcache_mem_bus_tmp_if)
    );
`else
    assign flush_done_w[0] = 1'b1;
`endif // VX_CFG_EXT_TEX_ENABLE

    ///////////////////////////////////////////////////////////////////////////
    // RTU — socket-resident ray-traversal units + private rtcache
    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_RTU_ENABLE
    localparam RTU_CORES_PER_RTU = NC / `VX_CFG_NUM_RTU_CORES;
    localparam RTU_SRC_WIDTH     = `UP(`ARB_SEL_BITS(RTU_CORES_PER_RTU, 1));

    VX_rtu_bus_if #(
        .NUM_LANES (SFU_LANES),
        .TAG_WIDTH (RTU_REQ_TAG_WIDTH),
        .SRC_WIDTH (1)
    ) per_core_rtu_bus_if[NC]();

    for (genvar c = 0; c < NC; ++c) begin : g_rtu_bind
        assign per_core_rtu_bus_if[c].arm_valid            = rtu_arm_valid[c];
        assign per_core_rtu_bus_if[c].arm_data.src         = '0;
        assign per_core_rtu_bus_if[c].arm_data.wid         = rtu_arm_wid[c];
        assign per_core_rtu_bus_if[c].arm_data.mask        = rtu_arm_mask[c];
        assign per_core_rtu_bus_if[c].arm_data.scene_base  = rtu_arm_scene_base[c];
        assign per_core_rtu_bus_if[c].arm_data.flags       = rtu_arm_flags[c];
        assign per_core_rtu_bus_if[c].arm_data.cull_mask   = rtu_arm_cull_mask[c];
        assign per_core_rtu_bus_if[c].arm_data.payload_ptr = rtu_arm_payload_ptr[c];
        assign per_core_rtu_bus_if[c].arm_data.tag         = rtu_arm_tag[c];
        assign rtu_arm_ready[c] = per_core_rtu_bus_if[c].arm_ready;

        assign per_core_rtu_bus_if[c].req_valid          = rtu_req_valid[c];
        assign per_core_rtu_bus_if[c].req_data.kind      = rtu_req_kind[c];
        assign per_core_rtu_bus_if[c].req_data.src       = '0;
        assign per_core_rtu_bus_if[c].req_data.wid       = rtu_req_wid[c];
        assign per_core_rtu_bus_if[c].req_data.data      = rtu_req_data[c];
        assign per_core_rtu_bus_if[c].req_data.cb_action = rtu_req_cb_action[c];
        assign rtu_req_ready[c] = per_core_rtu_bus_if[c].req_ready;

        assign rtu_win_valid[c]   = per_core_rtu_bus_if[c].win_valid;
        assign rtu_win_is_cand[c] = per_core_rtu_bus_if[c].win_data.is_cand;
        assign rtu_win_wid[c]     = per_core_rtu_bus_if[c].win_data.wid;
        assign rtu_win_slot[c]    = per_core_rtu_bus_if[c].win_data.slot;
        assign rtu_win_mask[c]    = per_core_rtu_bus_if[c].win_data.mask;
        assign rtu_win_data[c]    = per_core_rtu_bus_if[c].win_data.data;
        assign rtu_win_tag[c]     = per_core_rtu_bus_if[c].win_data.tag;
        assign per_core_rtu_bus_if[c].win_ready = rtu_win_ready[c];
    end

    VX_rtu_bus_if #(
        .NUM_LANES (SFU_LANES),
        .TAG_WIDTH (RTU_REQ_ARB1_TAG_WIDTH),
        .SRC_WIDTH (RTU_SRC_WIDTH)
    ) rtu_bus_if [`VX_CFG_NUM_RTU_CORES] ();

    VX_rtu_bus_arb #(
        .NUM_INPUTS  (NC),
        .NUM_LANES   (SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_RTU_CORES),
        .TAG_WIDTH   (RTU_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_ARM ((NC != `VX_CFG_NUM_RTU_CORES) ? 3 : 0),
        .OUT_BUF_REQ ((NC != `VX_CFG_NUM_RTU_CORES) ? 3 : 0),
        .OUT_BUF_WIN ((NC != `VX_CFG_NUM_RTU_CORES) ? 3 : 0)
    ) rtu_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_rtu_bus_if),
        .bus_out_if (rtu_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_WORD_SIZE),
        .TAG_WIDTH (RTCACHE_TAG_WIDTH)
    ) rtcache_bus_if [`VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS] ();

    for (genvar i = 0; i < `VX_CFG_NUM_RTU_CORES; ++i) begin : g_rtu_core
        VX_rtu_core #(
            .INSTANCE_ID     (`SFORMATF(("%s-rtu%0d", INSTANCE_ID, i))),
            .NUM_LANES       (SFU_LANES),
            .NUM_SRCS        (RTU_CORES_PER_RTU),
            .TAG_WIDTH       (RTU_REQ_ARB1_TAG_WIDTH),
            .CACHE_DATA_SIZE (RTCACHE_WORD_SIZE),
            .CACHE_TAG_WIDTH (RTCACHE_TAG_WIDTH)
        ) rtu_core (
            .clk          (clk),
            .reset        (reset),
            .rtu_bus_if   (rtu_bus_if[i]),
            .cache_bus_if (rtcache_bus_if[i * RTCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_LINE_SIZE),
        .TAG_WIDTH (RTCACHE_MEM_TAG_WIDTH)
    ) rtcache_mem_bus_tmp_if [RTCACHE_MEM_PORTS] ();

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_WORD_SIZE),
        .TAG_WIDTH (RTCACHE_BUS_TAG_WIDTH)
    ) rtcache_flushable_bus_if [`VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS] ();

    VX_dcr_flush_if rtcache_flush_if();
    assign rtcache_flush_if.req = flush_req;
    assign flush_done_w[1] = rtcache_flush_if.done;

    VX_dcr_flush #(
        .WORD_SIZE   (RTCACHE_WORD_SIZE),
        .TAG_WIDTH   (RTCACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3)
    ) rtcache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (rtcache_flush_if),
        .core_bus_if  (rtcache_bus_if[0]),
        .cache_bus_if (rtcache_flushable_bus_if[0])
    );

    for (genvar i = 1; i < `VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS; ++i) begin : g_rtcache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (rtcache_flushable_bus_if[i], rtcache_bus_if[i],
                                  RTCACHE_BUS_TAG_WIDTH, RTCACHE_TAG_WIDTH, 0);
    end

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-rtcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_RTCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_RTU_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_RTCACHE_SIZE),
        .LINE_SIZE      (RTCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_RTCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_RTCACHE_NUM_WAYS),
        .WORD_SIZE      (RTCACHE_WORD_SIZE),
        .NUM_REQS       (RTCACHE_NUM_REQS),
        .MEM_PORTS      (RTCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_RTCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_RTCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_RTCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_RTCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RTCACHE_BUS_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3)
    ) rtcache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (rtcache_flushable_bus_if),
        .mem_bus_if     (rtcache_mem_bus_tmp_if)
    );
`else
    assign flush_done_w[1] = 1'b1;
`endif // VX_CFG_EXT_RTU_ENABLE

    ///////////////////////////////////////////////////////////////////////////
    // Socket memory arbiter: {icache, dcache, tcache, rtcache, DXA} -> trunk
    ///////////////////////////////////////////////////////////////////////////

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

            // icache slot (read-only stream from the ports)
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_valid = ic_mem_req_valid;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.rw     = 1'b0;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.byteen = '1;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.addr   = ic_mem_req_addr;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.attr   = '0;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.data   = '0;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_data.tag    = ic_mem_req_tag;
            assign ic_mem_req_ready = socket_arb_in_if[ICACHE_MEM_ARB_IDX].req_ready;
            assign ic_mem_rsp_valid = socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_valid;
            assign ic_mem_rsp_data  = socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_data.data;
            assign ic_mem_rsp_tag   = socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_data.tag;
            assign socket_arb_in_if[ICACHE_MEM_ARB_IDX].rsp_ready = ic_mem_rsp_ready;

            // dcache slot (read/write; OM aperture stores enter here)
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_valid       = dc_mem_req_valid[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.rw     = dc_mem_req_rw[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.byteen = dc_mem_req_byteen[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.addr   = dc_mem_req_addr[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.attr   = dc_mem_req_attr[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.data   = dc_mem_req_data[0];
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_data.tag    = dc_mem_req_tag[0];
            assign dc_mem_req_ready[0] = socket_arb_in_if[DCACHE_MEM_ARB_IDX].req_ready;
            assign dc_mem_rsp_valid[0] = socket_arb_in_if[DCACHE_MEM_ARB_IDX].rsp_valid;
            assign dc_mem_rsp_data[0]  = socket_arb_in_if[DCACHE_MEM_ARB_IDX].rsp_data.data;
            assign dc_mem_rsp_tag[0]   = socket_arb_in_if[DCACHE_MEM_ARB_IDX].rsp_data.tag;
            assign socket_arb_in_if[DCACHE_MEM_ARB_IDX].rsp_ready = dc_mem_rsp_ready[0];

        `ifdef VX_CFG_EXT_TEX_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[TCACHE_MEM_ARB_IDX], tcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_RTU_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[RTCACHE_MEM_ARB_IDX], rtcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, RTCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_DXA_ENABLE
            // DXA gmem slot (read-only tile fetches from the ports)
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_valid       = dxa_mem_req_valid;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.rw     = 1'b0;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.byteen = '1;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.addr   = dxa_mem_req_addr;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.attr   = '0;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.data   = '0;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data.tag    = dxa_mem_req_tag;
            assign dxa_mem_req_ready = socket_arb_in_if[DXA_MEM_ARB_IDX].req_ready;
            assign dxa_mem_rsp_valid = socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_valid;
            assign dxa_mem_rsp_data  = socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_data.data;
            assign dxa_mem_rsp_tag   = socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_data.tag;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_ready = dxa_mem_rsp_ready;
        `else
            `UNUSED_VAR (dxa_mem_req_valid)
            `UNUSED_VAR (dxa_mem_req_addr)
            `UNUSED_VAR (dxa_mem_req_tag)
            `UNUSED_VAR (dxa_mem_rsp_ready)
            assign dxa_mem_req_ready = 1'b0;
            assign dxa_mem_rsp_valid = 1'b0;
            assign dxa_mem_rsp_data  = '0;
            assign dxa_mem_rsp_tag   = '0;
        `endif

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
            // Extra dcache banks pass straight through, as VX_socket does.
            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
            ) dc_bank_bus_if();

            assign dc_bank_bus_if.req_valid       = dc_mem_req_valid[i];
            assign dc_bank_bus_if.req_data.rw     = dc_mem_req_rw[i];
            assign dc_bank_bus_if.req_data.byteen = dc_mem_req_byteen[i];
            assign dc_bank_bus_if.req_data.addr   = dc_mem_req_addr[i];
            assign dc_bank_bus_if.req_data.attr   = dc_mem_req_attr[i];
            assign dc_bank_bus_if.req_data.data   = dc_mem_req_data[i];
            assign dc_bank_bus_if.req_data.tag    = dc_mem_req_tag[i];
            assign dc_mem_req_ready[i] = dc_bank_bus_if.req_ready;
            assign dc_mem_rsp_valid[i] = dc_bank_bus_if.rsp_valid;
            assign dc_mem_rsp_data[i]  = dc_bank_bus_if.rsp_data.data;
            assign dc_mem_rsp_tag[i]   = dc_bank_bus_if.rsp_data.tag;
            assign dc_bank_bus_if.rsp_ready = dc_mem_rsp_ready[i];

            `ASSIGN_VX_MEM_BUS_IF_EX (socket_mem_bus_if[i], dc_bank_bus_if, SOCKET_MEM_ARB_TAG_WIDTH, SOCKET_MEM_TAG_WIDTH, UUID_WIDTH);
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // Cluster graphics container: raster + rcache, OM trunk stage + ocache
    ///////////////////////////////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) l2_core_if [L2_NUM_REQS] ();

`ifdef EXT_GFX_CLUSTER_ENABLE
    VX_dcr_bus_if gfx_dcr_bus_if();
    assign gfx_dcr_bus_if.req_valid = dcr_req_valid;
    assign gfx_dcr_bus_if.req_data  = dcr_req_data;
    assign dcr_rsp_valid            = gfx_dcr_bus_if.rsp_valid;
    `UNUSED_VAR (gfx_dcr_bus_if.rsp_data)

    VX_dcr_flush_if cluster_flush_if();
    assign cluster_flush_if.req = flush_req;
    assign flush_done_w[2] = cluster_flush_if.done;

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_kmu_bus_if raster_kmu_bus_if[1]();
    assign kmu_valid = raster_kmu_bus_if[0].valid;
    assign kmu_kind  = raster_kmu_bus_if[0].kind;
    assign kmu_eop   = raster_kmu_bus_if[0].eop;
    assign kmu_dest  = raster_kmu_bus_if[0].dest;
    assign kmu_data  = raster_kmu_bus_if[0].data;
    assign raster_kmu_bus_if[0].ready = kmu_ready;

    VX_raster_launch_if raster_launch_if[1]();
    assign raster_launch_if[0].valid = raster_launch_valid;
    assign raster_launch_ready = raster_launch_if[0].ready;

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) rcache_l2_bus_if();
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) gfx_l2_out_bus_if [L2_SOCKET_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) ocache_l2_bus_if();
`endif

    VX_graphics #(
        .CLUSTER_ID (0)
    ) graphics (
        .clk        (clk),
        .reset      (reset),
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_kmu_bus_if (raster_kmu_bus_if),
        .rcache_mem_bus_if (rcache_l2_bus_if),
        .raster_launch_if  (raster_launch_if),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .socket_mem_bus_if (socket_mem_bus_if),
        .l2_out_bus_if     (gfx_l2_out_bus_if),
        .ocache_mem_bus_if (ocache_l2_bus_if),
    `endif
        .dcr_bus_if        (gfx_dcr_bus_if),
        .cluster_flush_if  (cluster_flush_if),
        .busy              (busy)
    );

`ifdef VX_CFG_EXT_OM_ENABLE
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (l2_core_if[i], gfx_l2_out_bus_if[i]);
    end
`else
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (l2_core_if[i], socket_mem_bus_if[i]);
    end
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (l2_core_if[L2_GFX_RASTER_IDX], rcache_l2_bus_if);
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    `ASSIGN_VX_MEM_BUS_IF (l2_core_if[L2_GFX_OM_IDX], ocache_l2_bus_if);
`endif

`else // !EXT_GFX_CLUSTER_ENABLE
    // No cluster-resident graphics: the trunk connects straight to the L2.
    assign dcr_rsp_valid   = 1'b0;
    assign flush_done_w[2] = 1'b1;
    assign busy            = 1'b0;
    `UNUSED_VAR (dcr_req_valid)
    `UNUSED_VAR (dcr_req_data)

    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_socket_l2
        `ASSIGN_VX_MEM_BUS_IF (l2_core_if[i], socket_mem_bus_if[i]);
    end
`endif // EXT_GFX_CLUSTER_ENABLE

    assign flush_done = & flush_done_w;

    ///////////////////////////////////////////////////////////////////////////
    // Shared L2 (PASSTHRU wires when VX_CFG_L2_ENABLE is off)
    ///////////////////////////////////////////////////////////////////////////

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

endmodule
