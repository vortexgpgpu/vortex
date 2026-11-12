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

// One ALU block: the integer and mul/div datapaths behind a PE switch.
module VX_alu_block import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING BLOCK_STR   = "",
    parameter NUM_LANES  = 1,
    parameter PARTIAL_BW = 0
) (
    input wire              clk,
    input wire              reset,

    input wire [`LOG2UP(`VX_CFG_NUM_ALU_BLOCKS)-1:0] block_idx,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,
    VX_branch_ctl_if.master branch_ctl_if
);
    localparam PE_COUNT     = 1 + `VX_CFG_EXT_M_ENABLED;
    localparam PE_SEL_BITS  = `CLOG2(PE_COUNT);
    localparam PE_IDX_INT   = 0;
    localparam PE_IDX_MDV   = PE_IDX_INT + `VX_CFG_EXT_M_ENABLED;

    VX_execute_if #(
        .data_t (alu_execute_t)
    ) pe_execute_if[PE_COUNT]();

    VX_result_if#(
        .data_t (alu_result_t)
    ) pe_result_if[PE_COUNT]();

    reg [`UP(PE_SEL_BITS)-1:0] pe_select;
    always @(*) begin
        pe_select = PE_IDX_INT;
        if (`VX_CFG_EXT_M_ENABLED && (execute_if.data.op_args.alu.xtype == ALU_TYPE_MULDIV)) begin
            pe_select = PE_IDX_MDV;
        end
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
        .execute_in_if  (execute_if),
        .result_out_if  (result_if),
        .execute_out_if (pe_execute_if),
        .result_in_if   (pe_result_if)
    );

    VX_alu_int #(
        .INSTANCE_ID (`SFORMATF(("%s-int%s", INSTANCE_ID, BLOCK_STR))),
        .NUM_LANES   (NUM_LANES)
    ) alu_int (
        .clk           (clk),
        .reset         (reset),
        .block_idx     (block_idx),
        .execute_if    (pe_execute_if[PE_IDX_INT]),
        .branch_ctl_if (branch_ctl_if),
        .result_if     (pe_result_if[PE_IDX_INT])
    );

`ifdef VX_CFG_EXT_M_ENABLE
    VX_alu_muldiv #(
        .INSTANCE_ID (`SFORMATF(("%s-muldiv%s", INSTANCE_ID, BLOCK_STR))),
        .NUM_LANES   (NUM_LANES)
    ) muldiv_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_MDV]),
        .result_if  (pe_result_if[PE_IDX_MDV])
    );
`endif

endmodule
