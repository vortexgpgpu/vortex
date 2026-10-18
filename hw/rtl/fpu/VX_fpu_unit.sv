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

`include "VX_fpu_define.vh"

module VX_fpu_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH],
    VX_fpu_csr_if.master    fpu_csr_if[`VX_CFG_NUM_FPU_BLOCKS]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE = `VX_CFG_NUM_FPU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_FPU_LANES;
    localparam PARTIAL_BW = (BLOCK_SIZE != `VX_CFG_ISSUE_WIDTH) || (NUM_LANES != `VX_CFG_SIMD_WIDTH);

    VX_execute_if #(
        .data_t (fpu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  ('0),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (fpu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        VX_fpu_block #(
            .INSTANCE_ID (INSTANCE_ID),
            .NUM_LANES   (NUM_LANES),
            .PARTIAL_BW  (PARTIAL_BW)
        ) fpu_block (
            .clk        (clk),
            .reset      (reset),
            .block_idx  (`LOG2UP(BLOCK_SIZE)'(block_idx)),
            .execute_if (per_block_execute_if[block_idx]),
            .result_if  (per_block_result_if[block_idx]),
            .fpu_csr_if (fpu_csr_if[block_idx])
        );
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
