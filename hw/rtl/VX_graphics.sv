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

module VX_graphics import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef EXT_RASTER_ENABLE
    VX_raster_bus_if.master per_socket_raster_bus_if [`NUM_SOCKETS],
    VX_mem_bus_if.master    rcache_mem_bus_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.master perf_raster_if [`NUM_SOCKETS],
    output cache_perf_t     perf_rcache,
`endif
`endif

`ifdef EXT_TEX_ENABLE
    VX_tex_bus_if.slave     per_socket_tex_bus_if [`NUM_SOCKETS],
    VX_mem_bus_if.master    tcache_mem_bus_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.master   perf_tex_if [`NUM_SOCKETS],
    output cache_perf_t     perf_tcache,
`endif
`endif

`ifdef EXT_OM_ENABLE
    VX_om_bus_if.slave      per_socket_om_bus_if [`NUM_SOCKETS],
    VX_mem_bus_if.master    ocache_mem_bus_if,
`ifdef PERF_ENABLE
    VX_om_perf_if.master    perf_om_if [`NUM_SOCKETS],
    output cache_perf_t     perf_ocache,
`endif
`endif

    // DCRs
    VX_dcr_bus_if.slave     dcr_bus_if
);

    `UNUSED_PARAM (CLUSTER_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

`ifdef EXT_RASTER_ENABLE

`ifdef PERF_ENABLE
    VX_raster_perf_if perf_raster_unit_if[`NUM_RASTER_UNITS]();
    `PERF_RASTER_ADD (perf_raster_if, perf_raster_unit_if, `NUM_SOCKETS, `NUM_RASTER_UNITS)
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_bus_if[`NUM_RASTER_UNITS * RCACHE_NUM_REQS]();

    VX_raster_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) raster_bus_if[`NUM_RASTER_UNITS]();

    VX_dcr_bus_if raster_dcr_bus_tmp_if();
    assign raster_dcr_bus_tmp_if.write_valid = dcr_bus_if.write_valid && (dcr_bus_if.write_addr >= `VX_DCR_RASTER_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_RASTER_STATE_END);
    assign raster_dcr_bus_tmp_if.write_addr  = dcr_bus_if.write_addr;
    assign raster_dcr_bus_tmp_if.write_data  = dcr_bus_if.write_data;

    VX_dcr_bus_if raster_dcr_bus_if();
    `BUFFER_DCR_BUS_IF (raster_dcr_bus_if, raster_dcr_bus_tmp_if, 1);

    // Generate all raster units
    for (genvar i = 0; i < `NUM_RASTER_UNITS; ++i) begin

        `RESET_RELAY (raster_reset, reset);

        VX_raster_unit #(
            .INSTANCE_ID     ($sformatf("cluster%0d-raster%0d", CLUSTER_ID, i)),
            .INSTANCE_IDX    (CLUSTER_ID * `NUM_RASTER_UNITS + i),
            .NUM_INSTANCES   (`NUM_CLUSTERS * `NUM_RASTER_UNITS),
            .NUM_SLICES      (`RASTER_NUM_SLICES),
            .TILE_LOGSIZE    (`RASTER_TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (`RASTER_BLOCK_LOGSIZE),
            .MEM_FIFO_DEPTH  (`RASTER_MEM_FIFO_DEPTH),
            .QUAD_FIFO_DEPTH (`RASTER_QUAD_FIFO_DEPTH),
            .OUTPUT_QUADS    (`NUM_SFU_LANES)
        ) raster_unit (
            `SCOPE_IO_BIND (i)
            .clk           (clk),
            .reset         (raster_reset),
        `ifdef PERF_ENABLE
            .perf_raster_if(perf_raster_unit_if[i]),
        `endif
            .dcr_bus_if    (raster_dcr_bus_if),
            .raster_bus_if (raster_bus_if[i]),
            .cache_bus_if  (rcache_bus_if[i * RCACHE_NUM_REQS +: RCACHE_NUM_REQS])
        );
    end

    `RESET_RELAY (raster_arb_reset, reset);

    VX_raster_arb #(
        .NUM_INPUTS  (`NUM_RASTER_UNITS),
        .NUM_LANES   (`NUM_SFU_LANES),
        .NUM_OUTPUTS (`NUM_SOCKETS),
        .ARBITER     ("R"),
        .OUT_BUF     ((`NUM_SOCKETS != `NUM_RASTER_UNITS) ? 2 : 0)
    ) raster_arb (
        .clk        (clk),
        .reset      (raster_arb_reset),
        .bus_in_if  (raster_bus_if),
        .bus_out_if (per_socket_raster_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_LINE_SIZE),
        .TAG_WIDTH (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_bus_tmp_if();

    `RESET_RELAY (rcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-rcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_RCACHES),
        .NUM_INPUTS     (`NUM_RASTER_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`RCACHE_SIZE),
        .LINE_SIZE      (RCACHE_LINE_SIZE),
        .NUM_BANKS      (`RCACHE_NUM_BANKS),
        .NUM_WAYS       (`RCACHE_NUM_WAYS),
        .WORD_SIZE      (RCACHE_WORD_SIZE),
        .NUM_REQS       (RCACHE_NUM_REQS),
        .CRSQ_SIZE      (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`RCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) rcache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_rcache),
    `endif
        .clk            (clk),
        .reset          (rcache_reset),
        .core_bus_if    (rcache_bus_if),
        .mem_bus_if     (rcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_X (rcache_mem_bus_if, rcache_mem_bus_tmp_if, L2_TAG_WIDTH, RCACHE_MEM_TAG_WIDTH);

`endif

`ifdef EXT_TEX_ENABLE

`ifdef PERF_ENABLE
    VX_tex_perf_if perf_tex_unit_if[`NUM_TEX_UNITS]();
    `PERF_TEX_ADD (perf_tex_if, perf_tex_unit_if, `NUM_SOCKETS, `NUM_TEX_UNITS)
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_bus_if[`NUM_TEX_UNITS * TCACHE_NUM_REQS]();

    VX_tex_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES),
        .TAG_WIDTH (`TEX_REQ_ARB2_TAG_WIDTH)
    ) tex_bus_if[`NUM_TEX_UNITS]();

    `RESET_RELAY (tex_arb_reset, reset);

    VX_tex_arb #(
        .NUM_INPUTS   (`NUM_SOCKETS),
        .NUM_LANES    (`NUM_SFU_LANES),
        .NUM_OUTPUTS  (`NUM_TEX_UNITS),
        .TAG_WIDTH    (`TEX_REQ_ARB1_TAG_WIDTH),
        .ARBITER      ("R"),
        .OUT_BUF_REQ  ((`NUM_SOCKETS != `NUM_TEX_UNITS) ? 2 : 0)
    ) tex_arb (
        .clk        (clk),
        .reset      (tex_arb_reset),
        .bus_in_if  (per_socket_tex_bus_if),
        .bus_out_if (tex_bus_if)
    );

    VX_dcr_bus_if tex_dcr_bus_tmp_if();
    assign tex_dcr_bus_tmp_if.write_valid = dcr_bus_if.write_valid && (dcr_bus_if.write_addr >= `VX_DCR_TEX_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_TEX_STATE_END);
    assign tex_dcr_bus_tmp_if.write_addr  = dcr_bus_if.write_addr;
    assign tex_dcr_bus_tmp_if.write_data  = dcr_bus_if.write_data;

    VX_dcr_bus_if tex_dcr_bus_if();
    `BUFFER_DCR_BUS_IF (tex_dcr_bus_if, tex_dcr_bus_tmp_if, 1);

    // Generate all texture units
    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin

        `RESET_RELAY (tex_reset, reset);

        VX_tex_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-tex%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_SFU_LANES),
            .TAG_WIDTH   (`TEX_REQ_ARB2_TAG_WIDTH)
        ) tex_unit (
            .clk          (clk),
            .reset        (tex_reset),
        `ifdef PERF_ENABLE
            .perf_tex_if  (perf_tex_unit_if[i]),
        `endif
            .dcr_bus_if   (tex_dcr_bus_if),
            .tex_bus_if   (tex_bus_if[i]),
            .cache_bus_if (tcache_bus_if[i * TCACHE_NUM_REQS +: TCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_tmp_if();

    `RESET_RELAY (tcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-tcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_TCACHES),
        .NUM_INPUTS     (`NUM_TEX_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (`UUID_WIDTH),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) tcache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_tcache),
    `endif
        .clk            (clk),
        .reset          (tcache_reset),
        .core_bus_if    (tcache_bus_if),
        .mem_bus_if     (tcache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_X (tcache_mem_bus_if, tcache_mem_bus_tmp_if, L2_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH);

`endif

`ifdef EXT_OM_ENABLE

`ifdef PERF_ENABLE
    VX_om_perf_if perf_om_unit_if[`NUM_OM_UNITS]();
    `PERF_OM_ADD (perf_om_if, perf_om_unit_if, `NUM_SOCKETS, `NUM_OM_UNITS)
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_bus_if[`NUM_OM_UNITS * OCACHE_NUM_REQS]();

    VX_om_bus_if #(
        .NUM_LANES (`NUM_SFU_LANES)
    ) om_bus_if[`NUM_OM_UNITS]();

    `RESET_RELAY (om_arb_reset, reset);

    VX_om_arb #(
        .NUM_INPUTS  (`NUM_SOCKETS),
        .NUM_LANES   (`NUM_SFU_LANES),
        .NUM_OUTPUTS (`NUM_OM_UNITS),
        .ARBITER     ("R"),
        .OUT_BUF    ((`NUM_SOCKETS != `NUM_OM_UNITS) ? 2 : 0)
    ) om_arb (
        .clk        (clk),
        .reset      (om_arb_reset),
        .bus_in_if  (per_socket_om_bus_if),
        .bus_out_if (om_bus_if)
    );

    VX_dcr_bus_if om_dcr_bus_tmp_if();
    assign om_dcr_bus_tmp_if.write_valid = dcr_bus_if.write_valid && (dcr_bus_if.write_addr >= `VX_DCR_OM_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_OM_STATE_END);
    assign om_dcr_bus_tmp_if.write_addr  = dcr_bus_if.write_addr;
    assign om_dcr_bus_tmp_if.write_data  = dcr_bus_if.write_data;

    VX_dcr_bus_if om_dcr_bus_if();
    `BUFFER_DCR_BUS_IF (om_dcr_bus_if, om_dcr_bus_tmp_if, 1);

    // Generate all OM units
    for (genvar i = 0; i < `NUM_OM_UNITS; ++i) begin

        `RESET_RELAY (om_reset, reset);

        VX_om_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-om%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_SFU_LANES)
        ) om_unit (
            .clk           (clk),
            .reset         (om_reset),
        `ifdef PERF_ENABLE
            .perf_om_if    (perf_om_unit_if[i]),
        `endif
            .dcr_bus_if    (om_dcr_bus_if),
            .om_bus_if     (om_bus_if[i]),
            .cache_bus_if  (ocache_bus_if[i * OCACHE_NUM_REQS +: OCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_LINE_SIZE),
        .TAG_WIDTH (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_bus_tmp_if();

    `RESET_RELAY (ocache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-ocache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_OCACHES),
        .NUM_INPUTS     (`NUM_OM_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`OCACHE_SIZE),
        .LINE_SIZE      (OCACHE_LINE_SIZE),
        .NUM_BANKS      (`OCACHE_NUM_BANKS),
        .NUM_WAYS       (`OCACHE_NUM_WAYS),
        .WORD_SIZE      (OCACHE_WORD_SIZE),
        .NUM_REQS       (OCACHE_NUM_REQS),
        .CRSQ_SIZE      (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`OCACHE_MREQ_SIZE),
        .TAG_WIDTH      (OCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_WIDTH),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (2),
        .MEM_OUT_BUF    (2)
    ) ocache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_ocache),
    `endif
        .clk            (clk),
        .reset          (ocache_reset),

        .core_bus_if    (ocache_bus_if),
        .mem_bus_if     (ocache_mem_bus_tmp_if)
    );

    `ASSIGN_VX_MEM_BUS_IF_X (ocache_mem_bus_if, ocache_mem_bus_tmp_if, L2_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH);

`endif

endmodule
