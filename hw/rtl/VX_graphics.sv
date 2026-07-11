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

// Cluster-level wrapper that owns the shared RASTER and OM units
// and their associated caches (rcache / ocache).

`include "VX_define.vh"

module VX_graphics import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
`ifdef VX_CFG_EXT_RASTER_ENABLE
    output raster_perf_t    raster_perf,
    output cache_perf_t     rcache_perf,
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    output om_perf_t        om_perf,
    output cache_perf_t     ocache_perf,
`endif
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if.master per_socket_raster_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    rcache_mem_bus_if,
    // Delegated draw launch (device KMU → raster engines)
    VX_raster_launch_if.slave raster_launch_if[1],
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_om_bus_if.slave      per_socket_om_bus_if [NUM_SOCKETS],
    VX_mem_bus_if.master    ocache_mem_bus_if,
`endif

    // DCR (raw cluster-level slave; each unit's DCR slave filters by addr)
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Cluster-level flush trigger.
    VX_dcr_flush_if.slave   cluster_flush_if,

    // Producer busy — the raster engine signals frame drain out-of-band (it has
    // no in-band `done`); the cluster ORs this into the device busy aggregation.
    output wire             busy
);
    `UNUSED_PARAM (CLUSTER_ID)

    // Fan one DCR slave input out to one master per consumer unit. Each
    // unit's internal case-statement filters by DCR address range. The
    // VX_dcr_arb owns the rsp_valid/rsp_data signaling on dcr_bus_if so
    // VX_graphics doesn't drive them itself.
    localparam NUM_DCR_REQS = `VX_CFG_EXT_RASTER_ENABLED * `VX_CFG_NUM_RASTER_CORES
                            + `VX_CFG_EXT_OM_ENABLED * `VX_CFG_NUM_OM_CORES;

    VX_dcr_bus_if per_unit_dcr_bus_if [`UP(NUM_DCR_REQS)] ();

    VX_dcr_arb #(
        .NUM_REQS    (NUM_DCR_REQS),
        .REQ_OUT_BUF ((NUM_DCR_REQS > 1) ? 1 : 0)
    ) dcr_unit_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_bus_if),
        .bus_out_if (per_unit_dcr_bus_if)
    );

`ifdef VX_CFG_EXT_RASTER_ENABLE
    localparam DCR_RASTER_BASE = 0;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    localparam DCR_OM_BASE     = `VX_CFG_EXT_RASTER_ENABLED * `VX_CFG_NUM_RASTER_CORES;
`endif

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

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Early-Z committed-depth read ports: one OCACHE_NUM_REQS group per raster
    // engine, attached as extra ocache NUM_INPUTS in the OM block below so the
    // read is coherent with the OM's write-through depth stores.
    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_EARLYZ_TAG_WIDTH)
    ) earlyz_ocache_bus_if [`VX_CFG_NUM_RASTER_CORES * OCACHE_NUM_REQS] ();
`endif

`ifdef PERF_ENABLE
    VX_raster_perf_if per_core_raster_perf_if [`VX_CFG_NUM_RASTER_CORES] ();
`endif

    VX_raster_launch_if per_core_raster_launch_if[`VX_CFG_NUM_RASTER_CORES]();
    VX_raster_launch_fork #(
        .NUM_OUTPUTS (`VX_CFG_NUM_RASTER_CORES)
    ) raster_launch_fork (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (raster_launch_if[0]),
        .bus_out_if (per_core_raster_launch_if)
    );

    wire [`VX_CFG_NUM_RASTER_CORES-1:0] raster_busy_w;

    for (genvar i = 0; i < `VX_CFG_NUM_RASTER_CORES; ++i) begin : g_raster_core
        VX_raster_core #(
            .INSTANCE_ID     (`SFORMATF(("cluster%0d-raster%0d", CLUSTER_ID, i))),
            .INSTANCE_IDX    (CLUSTER_ID * `VX_CFG_NUM_RASTER_CORES + i),
            .NUM_INSTANCES   (`VX_CFG_NUM_CLUSTERS * `VX_CFG_NUM_RASTER_CORES),
            .NUM_SLICES      (`VX_CFG_RASTER_NUM_SLICES),
            // The front end's top-level walk unit is a coarse bin: bin_x/bin_y
            // arrive at BIN_LOG_SIZE granularity and VX_raster_te recursively
            // refines bin -> block -> quad, so the core's TILE_LOGSIZE must
            // carry BIN_LOG_SIZE to match. Note the te TILE_FIFO_DEPTH grows
            // as 4^(BIN-BLOCK).
            .TILE_LOGSIZE    (`VX_CFG_RASTER_BIN_LOG_SIZE),
            .BLOCK_LOGSIZE   (`VX_CFG_RASTER_BLOCK_LOG_SIZE),
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
            .launch_if       (per_core_raster_launch_if[i]),
            .raster_bus_if   (raster_bus_if[i]),
            .cache_bus_if    (rcache_bus_if[i * RCACHE_NUM_REQS +: RCACHE_NUM_REQS]),
        `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
            .earlyz_cache_bus_if (earlyz_ocache_bus_if[i * OCACHE_NUM_REQS +: OCACHE_NUM_REQS]),
        `endif
            .busy            (raster_busy_w[i])
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

    VX_raster_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_RASTER_CORES),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (NUM_SOCKETS),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `VX_CFG_NUM_RASTER_CORES) ? 3 : 0) // register only on fan-out (avoid double on 1:1 passthrough)
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

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_BUS_TAG_WIDTH)
    ) rcache_flushable_bus_if [`VX_CFG_NUM_RASTER_CORES * RCACHE_NUM_REQS] ();

    VX_dcr_flush_if rcache_flush_if();
    assign rcache_flush_if.req = cluster_flush_if.req;

    VX_dcr_flush #(
        .WORD_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (rcache_flush_if),
        .core_bus_if  (rcache_bus_if[0]),
        .cache_bus_if (rcache_flushable_bus_if[0])
    );

    for (genvar i = 1; i < `VX_CFG_NUM_RASTER_CORES * RCACHE_NUM_REQS; ++i) begin : g_rcache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (rcache_flushable_bus_if[i], rcache_bus_if[i],
                                  RCACHE_BUS_TAG_WIDTH, RCACHE_TAG_WIDTH, 0);
    end

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
        .TAG_WIDTH      (RCACHE_BUS_TAG_WIDTH),
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
        .core_bus_if    (rcache_flushable_bus_if),
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

    VX_om_bus_arb #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_OM_CORES),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_SOCKETS != `VX_CFG_NUM_OM_CORES) ? 3 : 0) // register only on fan-out (avoid double on 1:1 passthrough)
    ) om_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_om_bus_if),
        .bus_out_if (om_bus_if)
    );

`ifdef PERF_ENABLE
    VX_om_perf_if per_core_om_perf_if [`VX_CFG_NUM_OM_CORES] ();
`endif

    wire [`VX_CFG_NUM_OM_CORES-1:0] om_busy_w;

    for (genvar i = 0; i < `VX_CFG_NUM_OM_CORES; ++i) begin : g_om_core
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
            .cache_bus_if (ocache_bus_if[i * OCACHE_NUM_REQS +: OCACHE_NUM_REQS]),
            .busy         (om_busy_w[i])
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

    // Ocache core-side inputs: OM cores occupy the first NUM_OM_CORES input
    // groups; with early-Z, the raster engines' depth-read requesters occupy the
    // trailing NUM_RASTER_CORES input groups (coherent with OM write-through).
    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_BUS_TAG_WIDTH)
    ) ocache_flushable_bus_if [OCACHE_NUM_INPUTS * OCACHE_NUM_REQS] ();

    VX_dcr_flush_if ocache_flush_if();
    assign ocache_flush_if.req = cluster_flush_if.req;

    // OM-core flush chain carries the OM tag + 1 flush bit. When early-Z widens
    // the shared bus tag (a wider early-Z requester), the flushed OM bus is
    // zero-extended up to OCACHE_BUS_TAG_WIDTH below.
    localparam OCACHE_OM_FLUSH_TAG_WIDTH = OCACHE_TAG_WIDTH + 1;
    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_OM_FLUSH_TAG_WIDTH)
    ) ocache_om_flush_bus_if [1] ();

    VX_dcr_flush #(
        .WORD_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (ocache_flush_if),
        .core_bus_if  (ocache_bus_if[0]),
        .cache_bus_if (ocache_om_flush_bus_if[0])
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (ocache_flushable_bus_if[0], ocache_om_flush_bus_if[0],
                              OCACHE_BUS_TAG_WIDTH, OCACHE_OM_FLUSH_TAG_WIDTH, 0);

    for (genvar i = 1; i < `VX_CFG_NUM_OM_CORES * OCACHE_NUM_REQS; ++i) begin : g_ocache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (ocache_flushable_bus_if[i], ocache_bus_if[i],
                                  OCACHE_BUS_TAG_WIDTH, OCACHE_TAG_WIDTH, 0);
    end

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Attach the early-Z depth readers as the trailing ocache inputs. Each reader
    // presents OCACHE_EARLYZ_TAG_WIDTH; zero-extend to the shared bus tag width.
    for (genvar i = 0; i < `VX_CFG_NUM_RASTER_CORES * OCACHE_NUM_REQS; ++i) begin : g_earlyz_ocache_in
        localparam DST = `VX_CFG_NUM_OM_CORES * OCACHE_NUM_REQS + i;
        `ASSIGN_VX_MEM_BUS_IF_EX (ocache_flushable_bus_if[DST], earlyz_ocache_bus_if[i],
                                  OCACHE_BUS_TAG_WIDTH, OCACHE_EARLYZ_TAG_WIDTH, 0);
    end
`endif

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("cluster%0d-ocache", CLUSTER_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_OCACHES),
        .NUM_INPUTS     (OCACHE_NUM_INPUTS),
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
        .TAG_WIDTH      (OCACHE_BUS_TAG_WIDTH),
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
        .core_bus_if    (ocache_flushable_bus_if),
        .mem_bus_if     (ocache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_EX (ocache_mem_bus_if, ocache_mem_bus_tmp_if[0],
                              L2_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

`endif // VX_CFG_EXT_OM_ENABLE

    // ── Cluster-level gfx-cache flush done aggregation ─────────────────
    // Each gfx cache participates in flushing only if its extension is
    // compiled in; the inactive ones contribute a tied-1 so the AND still
    // resolves to the active set's combined done.
    wire rcache_flush_done;
    wire ocache_flush_done;
`ifdef VX_CFG_EXT_RASTER_ENABLE
    assign rcache_flush_done = rcache_flush_if.done;
`else
    assign rcache_flush_done = 1'b1;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    assign ocache_flush_done = ocache_flush_if.done;
`else
    assign ocache_flush_done = 1'b1;
`endif
    assign cluster_flush_if.done = rcache_flush_done & ocache_flush_done;

    // Producer busy = a raster engine still draining a frame (out-of-band
    // drain) or an OM core with fragments in flight (vx_om4 is fire-and-forget,
    // so nothing else holds the device busy until the ROP commits).
`ifdef VX_CFG_EXT_RASTER_ENABLE
    wire raster_busy_any = (| raster_busy_w);
`else
    wire raster_busy_any = 1'b0;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    wire om_busy_any = (| om_busy_w);
`else
    wire om_busy_any = 1'b0;
`endif
    assign busy = raster_busy_any || om_busy_any;

endmodule
