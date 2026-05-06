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

// Cluster-level wrapper that owns the shared TEX, RASTER and OM units
// and their associated caches (tcache / rcache / ocache).

`include "VX_define.vh"

module VX_graphics import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_bus_if.slave     per_socket_tex_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    tcache_mem_bus_if,
`endif

`ifdef EXT_RASTER_ENABLE
    VX_raster_bus_if.master per_socket_raster_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    rcache_mem_bus_if,
`endif

`ifdef EXT_OM_ENABLE
    VX_om_bus_if.slave      per_socket_om_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    ocache_mem_bus_if,
`endif

    // DCR (raw cluster-level slave; each unit's DCR slave filters by addr)
    VX_dcr_bus_if.slave     dcr_bus_if
);
    `UNUSED_PARAM (CLUSTER_ID)

    // Fan one DCR slave input out to one master per consumer unit. Each
    // unit's internal case-statement filters by DCR address range. The
    // VX_dcr_arb owns the rsp_valid/rsp_data signaling on dcr_bus_if so
    // VX_graphics doesn't drive them itself.
    localparam NUM_DCR_REQS = `EXT_TEX_ENABLED * `NUM_TEX_CORES
                            + `EXT_RASTER_ENABLED * `NUM_RASTER_CORES
                            + `EXT_OM_ENABLED * `NUM_OM_CORES;

    VX_dcr_bus_if per_unit_dcr_bus_if [NUM_DCR_REQS] ();

    VX_dcr_arb #(
        .NUM_REQS    (NUM_DCR_REQS),
        .REQ_OUT_BUF ((NUM_DCR_REQS > 1) ? 1 : 0)
    ) dcr_unit_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_bus_if),
        .bus_out_if (per_unit_dcr_bus_if)
    );

`ifdef EXT_TEX_ENABLE
    localparam DCR_TEX_BASE    = 0;
`endif
`ifdef EXT_RASTER_ENABLE
    localparam DCR_RASTER_BASE = `EXT_TEX_ENABLED * `NUM_TEX_CORES;
`endif
`ifdef EXT_OM_ENABLE
    localparam DCR_OM_BASE     = `EXT_TEX_ENABLED * `NUM_TEX_CORES
                               + `EXT_RASTER_ENABLED * `NUM_RASTER_CORES;
`endif

    /////////////////////////////////////////////////////////////////////////////
    // TEX
    /////////////////////////////////////////////////////////////////////////////

`ifdef EXT_TEX_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_bus_if [`NUM_TEX_CORES * TCACHE_NUM_REQS] ();

    VX_tex_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB2_TAG_WIDTH)
    ) tex_bus_if [`NUM_TEX_CORES] ();

    VX_tex_arb #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_LANES   (`NUM_SFU_LANES),
        .NUM_OUTPUTS (`NUM_TEX_CORES),
        .TAG_WIDTH   (TEX_REQ_ARB1_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_REQ ((NUM_SOCKETS != `NUM_TEX_CORES) ? 2 : 0)
    ) tex_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_tex_bus_if),
        .bus_out_if (tex_bus_if)
    );

    for (genvar i = 0; i < `NUM_TEX_CORES; ++i) begin : g_tex_unit
        VX_tex_core #(
            .INSTANCE_ID (`SFORMATF(("cluster%0d-tex%0d", CLUSTER_ID, i))),
            .NUM_LANES   (`NUM_SFU_LANES),
            .TAG_WIDTH   (TEX_REQ_ARB2_TAG_WIDTH)
        ) tex_core (
            .clk          (clk),
            .reset        (reset),
            .dcr_bus_if   (per_unit_dcr_bus_if[DCR_TEX_BASE + i]),
            .tex_bus_if   (tex_bus_if[i]),
            .cache_bus_if (tcache_bus_if[i * TCACHE_NUM_REQS +: TCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_tmp_if [TCACHE_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-tcache", CLUSTER_ID))),
        .NUM_UNITS      (`NUM_TCACHES),
        .NUM_INPUTS     (`NUM_TEX_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .MEM_PORTS      (TCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) tcache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (tcache_bus_if),
        .mem_bus_if     (tcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (tcache_mem_bus_if, tcache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // EXT_TEX_ENABLE

    /////////////////////////////////////////////////////////////////////////////
    // RASTER
    /////////////////////////////////////////////////////////////////////////////

`ifdef EXT_RASTER_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_bus_if [`NUM_RASTER_CORES * RCACHE_NUM_REQS] ();

    VX_raster_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) raster_bus_if [`NUM_RASTER_CORES] ();

    for (genvar i = 0; i < `NUM_RASTER_CORES; ++i) begin : g_raster_unit
        VX_raster_core #(
            .INSTANCE_ID     (`SFORMATF(("cluster%0d-raster%0d", CLUSTER_ID, i))),
            .INSTANCE_IDX    (CLUSTER_ID * `NUM_RASTER_CORES + i),
            .NUM_INSTANCES   (`NUM_CLUSTERS * `NUM_RASTER_CORES),
            .NUM_SLICES      (`RASTER_NUM_SLICES),
            .TILE_LOGSIZE    (`RASTER_TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (`RASTER_BLOCK_LOGSIZE),
            .MEM_FIFO_DEPTH  (`RASTER_MEM_FIFO_DEPTH),
            .QUAD_FIFO_DEPTH (`RASTER_QUAD_FIFO_DEPTH),
            .OUTPUT_QUADS    (`NUM_SFU_LANES)
        ) raster_core (
            .clk           (clk),
            .reset         (reset),
            .dcr_bus_if    (per_unit_dcr_bus_if[DCR_RASTER_BASE + i]),
            .raster_bus_if (raster_bus_if[i]),
            .cache_bus_if  (rcache_bus_if[i * RCACHE_NUM_REQS +: RCACHE_NUM_REQS])
        );
    end

    VX_raster_arb #(
        .NUM_INPUTS  (`NUM_RASTER_CORES),
        .NUM_LANES   (`NUM_SFU_LANES),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `NUM_RASTER_CORES) ? 2 : 0)
    ) raster_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (raster_bus_if),
        .bus_out_if (per_socket_raster_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_LINE_SIZE),
        .TAG_WIDTH (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_bus_tmp_if [RCACHE_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-rcache", CLUSTER_ID))),
        .NUM_UNITS      (`NUM_RCACHES),
        .NUM_INPUTS     (`NUM_RASTER_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`RCACHE_SIZE),
        .LINE_SIZE      (RCACHE_LINE_SIZE),
        .NUM_BANKS      (`RCACHE_NUM_BANKS),
        .NUM_WAYS       (`RCACHE_NUM_WAYS),
        .WORD_SIZE      (RCACHE_WORD_SIZE),
        .NUM_REQS       (RCACHE_NUM_REQS),
        .MEM_PORTS      (RCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`RCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) rcache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (rcache_bus_if),
        .mem_bus_if     (rcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (rcache_mem_bus_if, rcache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, RCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // EXT_RASTER_ENABLE

    /////////////////////////////////////////////////////////////////////////////
    // OM
    /////////////////////////////////////////////////////////////////////////////

`ifdef EXT_OM_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_bus_if [`NUM_OM_CORES * OCACHE_NUM_REQS] ();

    VX_om_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) om_bus_if [`NUM_OM_CORES] ();

    VX_om_arb #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_LANES   (`NUM_SFU_LANES),
        .NUM_OUTPUTS (`NUM_OM_CORES),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `NUM_OM_CORES) ? 2 : 0)
    ) om_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_om_bus_if),
        .bus_out_if (om_bus_if)
    );

    for (genvar i = 0; i < `NUM_OM_CORES; ++i) begin : g_om_unit
        VX_om_core #(
            .INSTANCE_ID (`SFORMATF(("cluster%0d-om%0d", CLUSTER_ID, i))),
            .NUM_LANES   (`NUM_SFU_LANES)
        ) om_core (
            .clk          (clk),
            .reset        (reset),
            .dcr_bus_if   (per_unit_dcr_bus_if[DCR_OM_BASE + i]),
            .om_bus_if    (om_bus_if[i]),
            .cache_bus_if (ocache_bus_if[i * OCACHE_NUM_REQS +: OCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_LINE_SIZE),
        .TAG_WIDTH (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_bus_tmp_if [OCACHE_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-ocache", CLUSTER_ID))),
        .NUM_UNITS      (`NUM_OCACHES),
        .NUM_INPUTS     (`NUM_OM_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`OCACHE_SIZE),
        .LINE_SIZE      (OCACHE_LINE_SIZE),
        .NUM_BANKS      (`OCACHE_NUM_BANKS),
        .NUM_WAYS       (`OCACHE_NUM_WAYS),
        .WORD_SIZE      (OCACHE_WORD_SIZE),
        .NUM_REQS       (OCACHE_NUM_REQS),
        .MEM_PORTS      (OCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`OCACHE_MREQ_SIZE),
        .TAG_WIDTH      (OCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) ocache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (ocache_bus_if),
        .mem_bus_if     (ocache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (ocache_mem_bus_if, ocache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // EXT_OM_ENABLE

endmodule
