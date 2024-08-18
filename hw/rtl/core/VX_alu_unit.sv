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

module VX_alu_unit #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH],
    VX_branch_ctl_if.master branch_ctl_if [`NUM_ALU_BLOCKS]
);

    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE   = `NUM_ALU_BLOCKS;
    localparam NUM_LANES    = `NUM_ALU_LANES;
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam RSP_ARB_DATAW= `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + NUM_LANES * `XLEN + PID_WIDTH + 1 + 1;
    localparam RSP_ARB_SIZE = 1 + `EXT_M_ENABLED;
    localparam PARTIAL_BW   = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `NUM_THREADS);

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 1 : 0)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_commit_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin

        `RESET_RELAY_EN (block_reset, reset,(BLOCK_SIZE > 1));

        wire is_muldiv_op = `EXT_M_ENABLED && (per_block_execute_if[block_idx].data.op_args.alu.xtype == `ALU_TYPE_MULDIV);

        VX_execute_if #(
            .NUM_LANES (NUM_LANES)
        ) int_execute_if();

        VX_commit_if #(
            .NUM_LANES (NUM_LANES)
        ) int_commit_if();

        assign int_execute_if.valid = per_block_execute_if[block_idx].valid && ~is_muldiv_op;
        assign int_execute_if.data = per_block_execute_if[block_idx].data;

        VX_alu_int #(
            .INSTANCE_ID ($sformatf("%s-int%0d", INSTANCE_ID, block_idx)),
            .BLOCK_IDX (block_idx),
            .NUM_LANES (NUM_LANES)
        ) alu_int (
            .clk        (clk),
            .reset      (block_reset),
            .execute_if (int_execute_if),
            .branch_ctl_if (branch_ctl_if[block_idx]),
            .commit_if  (int_commit_if)
        );

    `ifdef EXT_M_ENABLE

        VX_execute_if #(
            .NUM_LANES (NUM_LANES)
        ) muldiv_execute_if();

        VX_commit_if #(
            .NUM_LANES (NUM_LANES)
        ) muldiv_commit_if();

        assign muldiv_execute_if.valid = per_block_execute_if[block_idx].valid && is_muldiv_op;
        assign muldiv_execute_if.data = per_block_execute_if[block_idx].data;

        VX_alu_muldiv #(
            .INSTANCE_ID ($sformatf("%s-muldiv%0d", INSTANCE_ID, block_idx)),
            .NUM_LANES (NUM_LANES)
        ) muldiv_unit (
            .clk        (clk),
            .reset      (block_reset),
            .execute_if (muldiv_execute_if),
            .commit_if  (muldiv_commit_if)
        );

    `endif

        assign per_block_execute_if[block_idx].ready =
        `ifdef EXT_M_ENABLE
            is_muldiv_op ? muldiv_execute_if.ready :
        `endif
            int_execute_if.ready;

        // send response

        VX_stream_arb #(
            .NUM_INPUTS (RSP_ARB_SIZE),
            .DATAW      (RSP_ARB_DATAW),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3),
            .ARBITER    ("F")
        ) rsp_arb (
            .clk       (clk),
            .reset     (block_reset),
            .valid_in  ({
            `ifdef EXT_M_ENABLE
                muldiv_commit_if.valid,
            `endif
                int_commit_if.valid
            }),
            .ready_in  ({
            `ifdef EXT_M_ENABLE
                muldiv_commit_if.ready,
            `endif
                int_commit_if.ready
            }),
            .data_in   ({
            `ifdef EXT_M_ENABLE
                muldiv_commit_if.data,
            `endif
                int_commit_if.data
            }),
            .data_out  (per_block_commit_if[block_idx].data),
            .valid_out (per_block_commit_if[block_idx].valid),
            .ready_out (per_block_commit_if[block_idx].ready),
            `UNUSED_PIN (sel_out)
        );
    end

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) gather_unit (
        .clk           (clk),
        .reset         (reset),
        .commit_in_if  (per_block_commit_if),
        .commit_out_if (commit_if)
    );

endmodule
