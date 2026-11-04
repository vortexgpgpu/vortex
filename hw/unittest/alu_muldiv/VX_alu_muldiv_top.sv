// Copyright © 2019-2026
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

// VX_alu_muldiv behind flat ports: an operation's M opcode, W flag, operands
// and a tag (carried in the header's PC) in; the tag and the lanes' results
// out.
module VX_alu_muldiv_top import VX_gpu_pkg::*; #(
    parameter NUM_LANES = `VX_CFG_NUM_ALU_LANES
) (
    input wire                                   clk,
    input wire                                   reset,

    input wire                                   in_valid,
    output wire                                  in_ready,
    input wire [INST_M_BITS-1:0]                 in_op,
    input wire                                   in_is_w,
    input wire [31:0]                            in_tag,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] in_rs1,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] in_rs2,

    output wire                                  out_valid,
    input wire                                   out_ready,
    output wire [31:0]                           out_tag,
    output wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] out_data
);
    VX_execute_if #(.data_t (alu_execute_t)) execute_if();
    VX_result_if  #(.data_t (alu_result_t))  result_if();

    alu_execute_t exe;
    always @(*) begin
        exe = '0;
        exe.header.PC   = PC_BITS'(in_tag);
        exe.op_type     = INST_OP_BITS'(in_op);
        exe.op_args.alu.is_w = in_is_w;
        exe.rs1_data    = in_rs1;
        exe.rs2_data    = in_rs2;
    end

    // the unit reads only its own fields of the operation; a narrow PC keeps
    // only the tag's low bits
    `UNUSED_VAR (execute_if.data)
    `UNUSED_VAR (in_tag)

    assign execute_if.valid = in_valid;
    assign execute_if.data  = exe;
    assign in_ready         = execute_if.ready;

    VX_alu_muldiv #(
        .NUM_LANES (NUM_LANES)
    ) dut (
        .clk        (clk),
        .reset      (reset),
        .execute_if (execute_if),
        .result_if  (result_if)
    );

    assign out_valid       = result_if.valid;
    assign result_if.ready = out_ready;
    assign out_tag         = 32'(result_if.data.header.PC);
    assign out_data        = result_if.data.data;
    `UNUSED_VAR (result_if.data)

endmodule
