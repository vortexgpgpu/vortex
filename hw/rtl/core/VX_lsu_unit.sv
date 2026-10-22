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

module VX_lsu_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH],

`ifdef PERF_ENABLE
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] perf_rd_req_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] perf_wr_req_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] perf_rsp_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] coalescer_misses,
`endif

`ifdef TCU_META_ENABLE
    VX_lsu_sched_if.slave   tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
`endif

    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] sched_empty,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] store_empty,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] busy,

`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.master    lmem_bus_if [LSU_NUM_REQS],
`endif
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
);
    localparam BLOCK_SIZE = `VX_CFG_NUM_LSU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_LSU_LANES;

    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (lsu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  ('0),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (lsu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef VX_CFG_LMEM_ENABLE
        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) block_lmem_bus_if[NUM_LANES]();
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_lmem_bus_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_bus_if[block_idx * NUM_LANES + j], block_lmem_bus_if[j]);
        end
    `endif

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH)
        ) block_dcache_bus_if[DCACHE_CHANNELS]();
        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[block_idx * DCACHE_CHANNELS + j], block_dcache_bus_if[j]);
        end

        VX_lsu_block #(
            .INSTANCE_ID (`SFORMATF(("%s%0d", INSTANCE_ID, block_idx)))
        ) lsu_block(
            `SCOPE_IO_BIND  (block_idx)
            .clk              (clk),
            .reset            (reset),
            .core_id          (core_id),
            .execute_if       (per_block_execute_if[block_idx]),
            .result_if        (per_block_result_if[block_idx]),
        `ifdef PERF_ENABLE
            .perf_rd_req_fire (perf_rd_req_fire[block_idx]),
            .perf_wr_req_fire (perf_wr_req_fire[block_idx]),
            .perf_rsp_fire    (perf_rsp_fire[block_idx]),
            .coalescer_misses (coalescer_misses[block_idx]),
        `endif
        `ifdef TCU_META_ENABLE
            .tcu_mem_if       (tcu_mem_if[block_idx]),
        `endif
            .sched_empty      (sched_empty[block_idx]),
            .store_empty      (store_empty[block_idx]),
            .busy             (busy[block_idx]),
        `ifdef VX_CFG_LMEM_ENABLE
            .lmem_bus_if      (block_lmem_bus_if),
        `endif
            .dcache_bus_if    (block_dcache_bus_if)
        );
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
