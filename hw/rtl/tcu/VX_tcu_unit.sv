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

module VX_tcu_unit import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output tcu_perf_t       tcu_perf,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master     tcu_lmem_if,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory client connection to VX_lsu_scheduler at VX_core.
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_TCU_LANES;

    `STATIC_ASSERT (BLOCK_SIZE == `VX_CFG_ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_execute_t)
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
        .data_t (tcu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [BLOCK_SIZE-1:0] exec_valid, exec_ready;
    tcu_execute_t         exec_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data;
    wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data;
    wire [BLOCK_SIZE-1:0] tbuf_ready_eff;
`endif

`ifdef TCU_META_ENABLE
    wire [BLOCK_SIZE-1:0] agu_ld_valid, agu_ld_ready;
    tcu_execute_t         agu_ld_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] agu_result_valid, agu_result_ready;
    tcu_result_t          agu_result_data [BLOCK_SIZE];
    wire                  meta_wr_en;
    wire [NW_WIDTH-1:0]   meta_wr_wid;
    wire [4:0]            meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] meta_wr_data;
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef TCU_META_ENABLE
        wire block_meta_wr_en = meta_wr_en
                             && (NW_WIDTH'(meta_wr_wid % `VX_CFG_NUM_TCU_BLOCKS) == NW_WIDTH'(block_idx));
    `endif

        VX_tcu_block #(
            .INSTANCE_ID (INSTANCE_ID),
            .BLOCK_STR   (`SFORMATF(("%0d", block_idx)))
        ) tcu_block (
            `SCOPE_IO_BIND (block_idx)
            .clk              (clk),
            .reset            (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .exec_valid       (exec_valid[block_idx]),
            .exec_ready       (exec_ready[block_idx]),
            .exec_data        (exec_data[block_idx]),
            .tbuf_rs1_data    (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data    (tbuf_rs2_data[block_idx]),
            .tbuf_ready       (tbuf_ready_eff[block_idx]),
        `endif
        `ifdef TCU_META_ENABLE
            .agu_ld_valid     (agu_ld_valid[block_idx]),
            .agu_ld_data      (agu_ld_data[block_idx]),
            .agu_ld_ready     (agu_ld_ready[block_idx]),
            .agu_result_valid (agu_result_valid[block_idx]),
            .agu_result_data  (agu_result_data[block_idx]),
            .agu_result_ready (agu_result_ready[block_idx]),
            .meta_wr_en       (block_meta_wr_en),
            .meta_wr_wid      (meta_wr_wid),
            .meta_wr_idx      (meta_wr_idx),
            .meta_wr_data     (meta_wr_data),
        `endif
            .execute_if       (per_block_execute_if[block_idx]),
            .result_if        (per_block_result_if[block_idx])
        );
    end

    VX_tcu_shared #(
        .INSTANCE_ID (INSTANCE_ID),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) tcu_shared (
    `ifdef PERF_ENABLE
        .tcu_perf         (tcu_perf),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .exec_valid       (exec_valid),
        .exec_ready       (exec_ready),
        .exec_data        (exec_data),
        .tbuf_rs1_data    (tbuf_rs1_data),
        .tbuf_rs2_data    (tbuf_rs2_data),
        .tbuf_ready_eff   (tbuf_ready_eff),
        .tcu_lmem_if      (tcu_lmem_if),
    `endif
    `ifdef TCU_META_ENABLE
        .agu_ld_valid     (agu_ld_valid),
        .agu_ld_data      (agu_ld_data),
        .agu_ld_ready     (agu_ld_ready),
        .agu_result_valid (agu_result_valid),
        .agu_result_data  (agu_result_data),
        .agu_result_ready (agu_result_ready),
        .meta_wr_en       (meta_wr_en),
        .meta_wr_wid      (meta_wr_wid),
        .meta_wr_idx      (meta_wr_idx),
        .meta_wr_data     (meta_wr_data),
        .tcu_mem_if       (tcu_mem_if),
    `endif
        .clk              (clk),
        .reset            (reset)
    );

    // -----------------------------------------------------------------------
    // Lane gather
    // -----------------------------------------------------------------------

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
