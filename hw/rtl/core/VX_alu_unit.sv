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
    localparam PARTIAL_BW   = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `NUM_THREADS);
    localparam PE_COUNT     = 1 + `EXT_M_ENABLED;
    localparam PE_SEL_BITS  = `CLOG2(PE_COUNT);
    localparam PE_IDX_INT   = 0;
    localparam PE_IDX_MDV   = PE_IDX_INT + `EXT_M_ENABLED;

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_commit_if[BLOCK_SIZE]();

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_alus

        VX_execute_if #(
            .NUM_LANES (NUM_LANES)
        ) pe_execute_if[PE_COUNT]();

        VX_commit_if#(
            .NUM_LANES (NUM_LANES)
        ) pe_commit_if[PE_COUNT]();

        reg [`UP(PE_SEL_BITS)-1:0] pe_select;
        always @(*) begin
            pe_select = PE_IDX_INT;
            if (`EXT_M_ENABLED && (per_block_execute_if[block_idx].data.op_args.alu.xtype == `ALU_TYPE_MULDIV))
                pe_select = PE_IDX_MDV;
        end

        VX_pe_switch #(
            .PE_COUNT    (PE_COUNT),
            .NUM_LANES   (NUM_LANES),
            .ARBITER     ("R"),
            .REQ_OUT_BUF (0),
            .RSP_OUT_BUF (PARTIAL_BW ? 1 : 3)
        ) pe_switch (
            .clk            (clk),
            .reset          (reset),
            .pe_sel         (pe_select),
            .execute_in_if  (per_block_execute_if[block_idx]),
            .commit_out_if  (per_block_commit_if[block_idx]),
            .execute_out_if (pe_execute_if),
            .commit_in_if   (pe_commit_if)
        );

        VX_alu_int #(
            .INSTANCE_ID ($sformatf("%s-int%0d", INSTANCE_ID, block_idx)),
            .BLOCK_IDX (block_idx),
            .NUM_LANES (NUM_LANES)
        ) alu_int (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_INT]),
            .branch_ctl_if (branch_ctl_if[block_idx]),
            .commit_if  (pe_commit_if[PE_IDX_INT])
        );

    `ifdef EXT_M_ENABLE
        VX_alu_muldiv #(
            .INSTANCE_ID ($sformatf("%s-muldiv%0d", INSTANCE_ID, block_idx)),
            .NUM_LANES (NUM_LANES)
        ) muldiv_unit (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_MDV]),
            .commit_if  (pe_commit_if[PE_IDX_MDV])
        );
    `endif
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
