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

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `NUM_TCU_LANES;
    localparam PE_COUNT   = 2;

    `STATIC_ASSERT (BLOCK_SIZE == `ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_exe_t)
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

    VX_result_if #(
        .data_t (tcu_res_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks

        VX_execute_if #(
            .data_t (tcu_exe_t)
        ) pe_execute_if[PE_COUNT]();

        VX_result_if #(
            .data_t (tcu_res_t)
        ) pe_result_if[PE_COUNT]();

        VX_pe_switch #(
            .PE_COUNT    (PE_COUNT),
            .NUM_LANES   (NUM_LANES),
            .ARBITER     ("R"),
            .REQ_OUT_BUF (0),
            .RSP_OUT_BUF (3)
        ) pe_switch (
            .clk            (clk),
            .reset          (reset),
            .pe_sel         (per_block_execute_if[block_idx].data.op_args.tcu.fmt_s[3]),
            .execute_in_if  (per_block_execute_if[block_idx]),
            .result_out_if  (per_block_result_if[block_idx]),
            .execute_out_if (pe_execute_if),
            .result_in_if   (pe_result_if)
        );

        VX_tcu_fp #(
            .INSTANCE_ID (`SFORMATF(("%s-fp%0d", INSTANCE_ID, block_idx)))
        ) tcu_fp (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[0]),
            .result_if  (pe_result_if[0])
        );

        VX_tcu_int #(
            .INSTANCE_ID (`SFORMATF(("%s-int%0d", INSTANCE_ID, block_idx)))
        ) tcu_int (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[1]),
            .result_if  (pe_result_if[1])
        );
    end

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) gather_unit (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
