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
    output cache_perf_t     lmem_perf,
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

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : lmem_switches
        VX_lmem_switch #(
            .REQ0_OUT_BUF (3),
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

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_bus_if[LSU_NUM_REQS]();

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : lmem_adapters
        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_bus_tmp_if[`NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - `UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) lmem_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (lsu_lmem_if[i]),
            .mem_bus_if (lmem_bus_tmp_if)
        );

        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            `ASSIGN_VX_MEM_BUS_IF (lmem_bus_if[i * `NUM_LSU_LANES + j], lmem_bus_tmp_if[j]);
        end
    end

    VX_local_mem #(
        .INSTANCE_ID($sformatf("%s-lmem", INSTANCE_ID)),
        .SIZE       (1 << `LMEM_LOG_SIZE),
        .NUM_REQS   (LSU_NUM_REQS),
        .NUM_BANKS  (`LMEM_NUM_BANKS),
        .WORD_SIZE  (LSU_WORD_SIZE),
        .ADDR_WIDTH (LMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH),
        .TAG_WIDTH  (LSU_TAG_WIDTH),
        .OUT_BUF    (3)
    ) local_mem (
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE
        .lmem_perf  (lmem_perf),
    `endif
        .mem_bus_if (lmem_bus_if)
    );

`else

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif
    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_coalesced_if[`NUM_LSU_BLOCKS]();

    if (LSU_WORD_SIZE != DCACHE_WORD_SIZE) begin : coalescer_if

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : coalescers
            VX_mem_coalescer #(
                .INSTANCE_ID    ($sformatf("%s-coalescer%0d", INSTANCE_ID, i)),
                .NUM_REQS       (`NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .FLAGS_WIDTH    (`MEM_REQ_FLAGS_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (`UUID_WIDTH),
                .QUEUE_SIZE     (`LSUQ_OUT_SIZE)
            ) mem_coalescer (
                .clk            (clk),
                .reset          (reset),

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

    end else begin

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
            `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        end

    end

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin : dcache_adapters

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

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin
            `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j]);
        end

    end

endmodule
