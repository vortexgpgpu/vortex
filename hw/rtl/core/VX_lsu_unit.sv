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

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH],
    VX_lsu_mem_if.master    lsu_mem_if [`NUM_LSU_BLOCKS]
);
    localparam BLOCK_SIZE = `NUM_LSU_BLOCKS;
    localparam NUM_LANES  = `NUM_LSU_LANES;

`ifdef SCOPE
    `SCOPE_IO_SWITCH (BLOCK_SIZE);
`endif

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_commit_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : lsus
        VX_lsu_slice #(
            .INSTANCE_ID ($sformatf("%s%0d", INSTANCE_ID, block_idx))
        ) lsu_slice(
            `SCOPE_IO_BIND  (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (per_block_execute_if[block_idx]),
            .commit_if  (per_block_commit_if[block_idx]),
            .lsu_mem_if (lsu_mem_if[block_idx])
        );
    end

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) gather_unit (
        .clk           (clk),
        .reset         (reset),
        .commit_in_if  (per_block_commit_if),
        .commit_out_if (commit_if)
    );

endmodule
