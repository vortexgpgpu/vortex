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

`ifdef PERF_ENABLE
`ifdef VX_CFG_EXT_TEX_ENABLE
    output tex_perf_t       tex_perf,
    output cache_perf_t     tcache_perf,
`endif
`ifdef VX_CFG_EXT_RASTER_ENABLE
    output raster_perf_t    raster_perf,
    output cache_perf_t     rcache_perf,
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    output om_perf_t        om_perf,
    output cache_perf_t     ocache_perf,
`endif
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.slave     per_socket_tex_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    tcache_mem_bus_if,
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if.master per_socket_raster_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    rcache_mem_bus_if,
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
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
    localparam NUM_DCR_REQS = `VX_CFG_EXT_TEX_ENABLED * `VX_CFG_NUM_TEX_CORES
                            + `VX_CFG_EXT_RASTER_ENABLED * `VX_CFG_NUM_RASTER_CORES
                            + `VX_CFG_EXT_OM_ENABLED * `VX_CFG_NUM_OM_CORES;

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

`ifdef VX_CFG_EXT_TEX_ENABLE
    localparam DCR_TEX_BASE    = 0;
`endif
`ifdef VX_CFG_EXT_RASTER_ENABLE
    localparam DCR_RASTER_BASE = `VX_CFG_EXT_TEX_ENABLED * `VX_CFG_NUM_TEX_CORES;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    localparam DCR_OM_BASE     = `VX_CFG_EXT_TEX_ENABLED * `VX_CFG_NUM_TEX_CORES
                               + `VX_CFG_EXT_RASTER_ENABLED * `VX_CFG_NUM_RASTER_CORES;
`endif

    /////////////////////////////////////////////////////////////////////////////
    // TEX
    /////////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_TEX_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_bus_if [`VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS] ();

    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB2_TAG_WIDTH)
    ) tex_bus_if [`VX_CFG_NUM_TEX_CORES] ();

    VX_tex_arb #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_TEX_CORES),
        .TAG_WIDTH   (TEX_REQ_ARB1_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_REQ ((NUM_SOCKETS != `VX_CFG_NUM_TEX_CORES) ? 2 : 0)
    ) tex_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_tex_bus_if),
        .bus_out_if (tex_bus_if)
    );

`ifdef PERF_ENABLE
    VX_tex_perf_if per_core_tex_perf_if [`VX_CFG_NUM_TEX_CORES] ();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin : g_tex_unit
        VX_tex_core #(
            .INSTANCE_ID (`SFORMATF(("cluster%0d-tex%0d", CLUSTER_ID, i))),
            .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
            .TAG_WIDTH   (TEX_REQ_ARB2_TAG_WIDTH)
        ) tex_core (
            .clk          (clk),
            .reset        (reset),
        `ifdef PERF_ENABLE
            .perf_tex_if  (per_core_tex_perf_if[i]),
        `endif
            .dcr_bus_if   (per_unit_dcr_bus_if[DCR_TEX_BASE + i]),
            .tex_bus_if   (tex_bus_if[i]),
            .cache_bus_if (tcache_bus_if[i * TCACHE_NUM_REQS +: TCACHE_NUM_REQS])
        );
    end

`ifdef PERF_ENABLE
    // Sum per-core TEX counters across the cluster. Verilator forbids
    // dynamic indexing into an interface array, so first copy each interface
    // member into a packed wire array via a genvar, then sum.
    wire [`VX_CFG_NUM_TEX_CORES-1:0][PERF_CTR_BITS-1:0] tex_mr_w, tex_ml_w, tex_sc_w;
    for (genvar i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin : g_tex_perf_pack
        assign tex_mr_w[i] = per_core_tex_perf_if[i].mem_reads;
        assign tex_ml_w[i] = per_core_tex_perf_if[i].mem_latency;
        assign tex_sc_w[i] = per_core_tex_perf_if[i].stall_cycles;
    end
    tex_perf_t tex_perf_sum;
    always @(*) begin
        tex_perf_sum = '0;
        for (int i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin
            tex_perf_sum.mem_reads    = tex_perf_sum.mem_reads    + tex_mr_w[i];
            tex_perf_sum.mem_latency  = tex_perf_sum.mem_latency  + tex_ml_w[i];
            tex_perf_sum.stall_cycles = tex_perf_sum.stall_cycles + tex_sc_w[i];
        end
    end
    assign tex_perf = tex_perf_sum;
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_tmp_if [TCACHE_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-tcache", CLUSTER_ID))),
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
    `ifdef PERF_ENABLE
        .cache_perf     (tcache_perf),
    `endif
        .core_bus_if    (tcache_bus_if),
        .mem_bus_if     (tcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (tcache_mem_bus_if, tcache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // VX_CFG_EXT_TEX_ENABLE

    /////////////////////////////////////////////////////////////////////////////
    // RASTER
    /////////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_RASTER_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_bus_if [`VX_CFG_NUM_RASTER_CORES * RCACHE_NUM_REQS] ();

    VX_raster_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES)
    ) raster_bus_if [`VX_CFG_NUM_RASTER_CORES] ();

`ifdef PERF_ENABLE
    VX_raster_perf_if per_core_raster_perf_if [`VX_CFG_NUM_RASTER_CORES] ();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_RASTER_CORES; ++i) begin : g_raster_unit
        VX_raster_core #(
            .INSTANCE_ID     (`SFORMATF(("cluster%0d-raster%0d", CLUSTER_ID, i))),
            .INSTANCE_IDX    (CLUSTER_ID * `VX_CFG_NUM_RASTER_CORES + i),
            .NUM_INSTANCES   (`VX_CFG_NUM_CLUSTERS * `VX_CFG_NUM_RASTER_CORES),
            .NUM_SLICES      (`VX_CFG_RASTER_NUM_SLICES),
            .TILE_LOGSIZE    (`VX_CFG_RASTER_TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (`VX_CFG_RASTER_BLOCK_LOGSIZE),
            .MEM_FIFO_DEPTH  (`VX_CFG_RASTER_MEM_FIFO_DEPTH),
            .QUAD_FIFO_DEPTH (`VX_CFG_RASTER_QUAD_FIFO_DEPTH),
            .OUTPUT_QUADS    (`VX_CFG_NUM_SFU_LANES)
        ) raster_core (
            .clk             (clk),
            .reset           (reset),
        `ifdef PERF_ENABLE
            .perf_raster_if  (per_core_raster_perf_if[i]),
        `endif
            .dcr_bus_if      (per_unit_dcr_bus_if[DCR_RASTER_BASE + i]),
            .raster_bus_if   (raster_bus_if[i]),
            .cache_bus_if    (rcache_bus_if[i * RCACHE_NUM_REQS +: RCACHE_NUM_REQS])
        );
    end

`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_RASTER_CORES-1:0][PERF_CTR_BITS-1:0] ras_mr_w, ras_ml_w, ras_sc_w;
    for (genvar i = 0; i < `VX_CFG_NUM_RASTER_CORES; ++i) begin : g_ras_perf_pack
        assign ras_mr_w[i] = per_core_raster_perf_if[i].mem_reads;
        assign ras_ml_w[i] = per_core_raster_perf_if[i].mem_latency;
        assign ras_sc_w[i] = per_core_raster_perf_if[i].stall_cycles;
    end
    raster_perf_t raster_perf_sum;
    always @(*) begin
        raster_perf_sum = '0;
        for (int i = 0; i < `VX_CFG_NUM_RASTER_CORES; ++i) begin
            raster_perf_sum.mem_reads    = raster_perf_sum.mem_reads    + ras_mr_w[i];
            raster_perf_sum.mem_latency  = raster_perf_sum.mem_latency  + ras_ml_w[i];
            raster_perf_sum.stall_cycles = raster_perf_sum.stall_cycles + ras_sc_w[i];
        end
    end
    assign raster_perf = raster_perf_sum;
`endif

    VX_raster_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_RASTER_CORES),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `VX_CFG_NUM_RASTER_CORES) ? 2 : 0)
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
        .NUM_UNITS      (`VX_CFG_NUM_RCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_RASTER_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_RCACHE_SIZE),
        .LINE_SIZE      (RCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_RCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_RCACHE_NUM_WAYS),
        .WORD_SIZE      (RCACHE_WORD_SIZE),
        .NUM_REQS       (RCACHE_NUM_REQS),
        .MEM_PORTS      (RCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_RCACHE_MREQ_SIZE),
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
    `ifdef PERF_ENABLE
        .cache_perf     (rcache_perf),
    `endif
        .core_bus_if    (rcache_bus_if),
        .mem_bus_if     (rcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (rcache_mem_bus_if, rcache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, RCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // VX_CFG_EXT_RASTER_ENABLE

    /////////////////////////////////////////////////////////////////////////////
    // OM
    /////////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_OM_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_bus_if [`VX_CFG_NUM_OM_CORES * OCACHE_NUM_REQS] ();

    VX_om_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES)
    ) om_bus_if [`VX_CFG_NUM_OM_CORES] ();

    VX_om_arb #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_OM_CORES),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `VX_CFG_NUM_OM_CORES) ? 2 : 0)
    ) om_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_om_bus_if),
        .bus_out_if (om_bus_if)
    );

`ifdef PERF_ENABLE
    VX_om_perf_if per_core_om_perf_if [`VX_CFG_NUM_OM_CORES] ();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_OM_CORES; ++i) begin : g_om_unit
        VX_om_core #(
            .INSTANCE_ID (`SFORMATF(("cluster%0d-om%0d", CLUSTER_ID, i))),
            .NUM_LANES   (`VX_CFG_NUM_SFU_LANES)
        ) om_core (
            .clk          (clk),
            .reset        (reset),
        `ifdef PERF_ENABLE
            .perf_om_if   (per_core_om_perf_if[i]),
        `endif
            .dcr_bus_if   (per_unit_dcr_bus_if[DCR_OM_BASE + i]),
            .om_bus_if    (om_bus_if[i]),
            .cache_bus_if (ocache_bus_if[i * OCACHE_NUM_REQS +: OCACHE_NUM_REQS])
        );
    end

`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_OM_CORES-1:0][PERF_CTR_BITS-1:0] om_mr_w, om_mw_w, om_ml_w, om_sc_w;
    for (genvar i = 0; i < `VX_CFG_NUM_OM_CORES; ++i) begin : g_om_perf_pack
        assign om_mr_w[i] = per_core_om_perf_if[i].mem_reads;
        assign om_mw_w[i] = per_core_om_perf_if[i].mem_writes;
        assign om_ml_w[i] = per_core_om_perf_if[i].mem_latency;
        assign om_sc_w[i] = per_core_om_perf_if[i].stall_cycles;
    end
    om_perf_t om_perf_sum;
    always @(*) begin
        om_perf_sum = '0;
        for (int i = 0; i < `VX_CFG_NUM_OM_CORES; ++i) begin
            om_perf_sum.mem_reads    = om_perf_sum.mem_reads    + om_mr_w[i];
            om_perf_sum.mem_writes   = om_perf_sum.mem_writes   + om_mw_w[i];
            om_perf_sum.mem_latency  = om_perf_sum.mem_latency  + om_ml_w[i];
            om_perf_sum.stall_cycles = om_perf_sum.stall_cycles + om_sc_w[i];
        end
    end
    assign om_perf = om_perf_sum;
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_LINE_SIZE),
        .TAG_WIDTH (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_bus_tmp_if [OCACHE_MEM_PORTS] ();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-ocache", CLUSTER_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_OCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_OM_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_OCACHE_SIZE),
        .LINE_SIZE      (OCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_OCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_OCACHE_NUM_WAYS),
        .WORD_SIZE      (OCACHE_WORD_SIZE),
        .NUM_REQS       (OCACHE_NUM_REQS),
        .MEM_PORTS      (OCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_OCACHE_MREQ_SIZE),
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
    `ifdef PERF_ENABLE
        .cache_perf     (ocache_perf),
    `endif
        .core_bus_if    (ocache_bus_if),
        .mem_bus_if     (ocache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (ocache_mem_bus_if, ocache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // VX_CFG_EXT_OM_ENABLE

endmodule
