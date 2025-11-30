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

module VX_mem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output lmem_perf_t      lmem_perf,
    output coalescer_perf_t coalescer_perf,
`endif

    VX_lsu_mem_if.slave     lsu_mem_if [`NUM_LSU_BLOCKS],
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
);
    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if[`NUM_LSU_BLOCKS]();

`ifdef LMEM_ENABLE

    `STATIC_ASSERT(`IS_DIVISBLE((1 << `LMEM_LOG_SIZE), `MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`LMEM_BASE_ADDR % (1 << `LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_if[`NUM_LSU_BLOCKS]();

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : g_lmem_switches
        VX_lmem_switch #(
            .REQ0_OUT_BUF (1),
            .REQ1_OUT_BUF (0),
            .RSP_OUT_BUF  (1),
            .ARBITER      ("P")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    end

    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) lmem_arb_if[1]();

    VX_lsu_mem_arb #(
        .NUM_INPUTS (`NUM_LSU_BLOCKS),
        .NUM_OUTPUTS(1),
        .NUM_LANES  (`NUM_LSU_LANES),
        .DATA_SIZE  (LSU_WORD_SIZE),
        .TAG_WIDTH  (LSU_TAG_WIDTH),
        .TAG_SEL_IDX(0),
        .ARBITER    ("R"),
        .REQ_OUT_BUF(0),
        .RSP_OUT_BUF(2)
    ) lmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (lsu_lmem_if),
        .bus_out_if (lmem_arb_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) lmem_adapt_if[`NUM_LSU_LANES]();

    VX_lsu_adapter #(
        .NUM_LANES    (`NUM_LSU_LANES),
        .DATA_SIZE    (LSU_WORD_SIZE),
        .TAG_WIDTH    (LMEM_TAG_WIDTH),
        .TAG_SEL_BITS (LMEM_TAG_WIDTH - `UUID_WIDTH),
        .ARBITER      ("P"),
        .REQ_OUT_BUF  (3),
        .RSP_OUT_BUF  (0)
    ) lmem_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (lmem_arb_if[0]),
        .mem_bus_if (lmem_adapt_if)
    );

    VX_local_mem #(
        .INSTANCE_ID(`SFORMATF(("%s-lmem", INSTANCE_ID))),
        .SIZE       (1 << `LMEM_LOG_SIZE),
        .NUM_REQS   (`NUM_LSU_LANES),
        .NUM_BANKS  (`LMEM_NUM_BANKS),
        .WORD_SIZE  (LSU_WORD_SIZE),
        .ADDR_WIDTH (LMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH),
        .TAG_WIDTH  (LMEM_TAG_WIDTH),
        .OUT_BUF    (3)
    ) local_mem (
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE
        .lmem_perf  (lmem_perf),
    `endif
        .mem_bus_if (lmem_adapt_if)
    );

`else

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : g_lsu_dcache_if
        `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_coalesced_if[`NUM_LSU_BLOCKS]();

`ifdef PERF_ENABLE
    wire [`NUM_LSU_BLOCKS-1:0][`PERF_CTR_BITS-1:0] per_block_coalescer_misses;
    wire [`PERF_CTR_BITS-1:0] coalescer_misses;
    VX_reduce_tree #(
        .DATAW_IN (`PERF_CTR_BITS),
        .DATAW_OUT (`PERF_CTR_BITS),
        .N  (`NUM_LSU_BLOCKS),
        .OP ("+")
    ) coalescer_reduce (
        .data_in  (per_block_coalescer_misses),
        .data_out (coalescer_misses)
    );
    `BUFFER(coalescer_perf.misses, coalescer_misses);
`endif

    if ((`NUM_LSU_LANES > 1) && (LSU_WORD_SIZE != DCACHE_WORD_SIZE)) begin : g_enabled

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : g_coalescers
            VX_mem_coalescer #(
                .INSTANCE_ID    (`SFORMATF(("%s-coalescer%0d", INSTANCE_ID, i))),
                .NUM_REQS       (`NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .FLAGS_WIDTH    (`MEM_REQ_FLAGS_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (`UUID_WIDTH),
                .QUEUE_SIZE     (`LSUQ_OUT_SIZE),
                .PERF_CTR_BITS  (`PERF_CTR_BITS)
            ) mem_coalescer (
                .clk            (clk),
                .reset          (reset),

            `ifdef PERF_ENABLE
                .misses         (per_block_coalescer_misses[i]),
            `else
                `UNUSED_PIN (misses),
            `endif

                // Input request
                .in_req_valid   (lsu_dcache_if[i].req_valid),
                .in_req_mask    (lsu_dcache_if[i].req_data.mask),
                .in_req_rw      (lsu_dcache_if[i].req_data.rw),
                .in_req_byteen  (lsu_dcache_if[i].req_data.byteen),
                .in_req_addr    (lsu_dcache_if[i].req_data.addr),
                .in_req_flags   (lsu_dcache_if[i].req_data.flags),
                .in_req_data    (lsu_dcache_if[i].req_data.data),
                .in_req_tag     (lsu_dcache_if[i].req_data.tag),
                .in_req_ready   (lsu_dcache_if[i].req_ready),

                // Input response
                .in_rsp_valid   (lsu_dcache_if[i].rsp_valid),
                .in_rsp_mask    (lsu_dcache_if[i].rsp_data.mask),
                .in_rsp_data    (lsu_dcache_if[i].rsp_data.data),
                .in_rsp_tag     (lsu_dcache_if[i].rsp_data.tag),
                .in_rsp_ready   (lsu_dcache_if[i].rsp_ready),

                // Output request
                .out_req_valid  (dcache_coalesced_if[i].req_valid),
                .out_req_mask   (dcache_coalesced_if[i].req_data.mask),
                .out_req_rw     (dcache_coalesced_if[i].req_data.rw),
                .out_req_byteen (dcache_coalesced_if[i].req_data.byteen),
                .out_req_addr   (dcache_coalesced_if[i].req_data.addr),
                .out_req_flags  (dcache_coalesced_if[i].req_data.flags),
                .out_req_data   (dcache_coalesced_if[i].req_data.data),
                .out_req_tag    (dcache_coalesced_if[i].req_data.tag),
                .out_req_ready  (dcache_coalesced_if[i].req_ready),

                // Output response
                .out_rsp_valid  (dcache_coalesced_if[i].rsp_valid),
                .out_rsp_mask   (dcache_coalesced_if[i].rsp_data.mask),
                .out_rsp_data   (dcache_coalesced_if[i].rsp_data.data),
                .out_rsp_tag    (dcache_coalesced_if[i].rsp_data.tag),
                .out_rsp_ready  (dcache_coalesced_if[i].rsp_ready)
            );
        end

    end else begin : g_passthru

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : g_dcache_coalesced_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        `ifdef PERF_ENABLE
            assign per_block_coalescer_misses[i] = '0;
        `endif
        end

    end

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : g_dcache_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH)
        ) dcache_bus_tmp_if[DCACHE_CHANNELS]();

        VX_lsu_adapter #(
            .NUM_LANES    (DCACHE_CHANNELS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_TAG_WIDTH),
            .TAG_SEL_BITS (DCACHE_TAG_WIDTH - `UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (0),
            .RSP_OUT_BUF  (0)
        ) dcache_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (dcache_coalesced_if[i]),
            .mem_bus_if (dcache_bus_tmp_if)
        );

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j]);
        end

    end

endmodule
