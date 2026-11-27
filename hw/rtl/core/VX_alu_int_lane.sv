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

// One lane's shift, logic and WGATHER results of the integer ALU. Only the
// lane's group of four is wired in, and its position in the group is an
// input, so every lane is the same module.
module VX_alu_int_lane import VX_gpu_pkg::*; #(
    parameter NUM_LANES = 1,
    // lanes per WGATHER group
    parameter WG_SIZE   = (NUM_LANES < 4) ? NUM_LANES : 4
) (
    input wire [1:0]        wg_pos,

    // the op's variant bits, shared by all lanes
    input wire [1:0]        op,
    input wire              is_signed,
    input wire [1:0]        wg_src_offset,

    // the lane's operands
    input wire [`VX_CFG_XLEN-1:0] in1,
    input wire [`VX_CFG_XLEN-1:0] in2,
    input wire [`VX_CFG_XLEN-1:0] in2_imm,

    // WGATHER sources: the lane's group and the last active lane
    input wire [WG_SIZE-1:0][`VX_CFG_XLEN-1:0] wg_in1,
    input wire [WG_SIZE-1:0][`VX_CFG_XLEN-1:0] wg_in2,
    input wire [WG_SIZE-1:0][`VX_CFG_XLEN-1:0] wg_in3,
    input wire [WG_SIZE-1:0] wg_tmask,
    input wire [`VX_CFG_XLEN-1:0] last_in1,
    input wire [`VX_CFG_XLEN-1:0] last_in2,
    input wire [`VX_CFG_XLEN-1:0] last_in3,

    output reg [`VX_CFG_XLEN-1:0]  shr_zic_result,
    output wire [`VX_CFG_XLEN-1:0] shr_result_w,
    output reg [`VX_CFG_XLEN-1:0]  msc_result,
    output wire [`VX_CFG_XLEN-1:0] msc_result_w,
    output wire [`VX_CFG_XLEN-1:0] wgather_result
);
    localparam SHIFT_IMM_BITS = `CLOG2(`VX_CFG_XLEN);
    localparam WG_BITS        = `CLOG2(WG_SIZE);

    // SRL, SRA, CZERO
    wire [`VX_CFG_XLEN:0] shr_in1 = {is_signed && in1[`VX_CFG_XLEN-1], in1};
    always @(*) begin
        case (op)
        `ifdef VX_CFG_EXT_ZICOND_ENABLE
            2'b10, 2'b11: begin // CZERO
                shr_zic_result = in1 & {`VX_CFG_XLEN{op[0] ^ (| in2)}};
            end
        `endif
            default: begin // SRL, SRA, SRLI, SRAI
                shr_zic_result = `VX_CFG_XLEN'($signed(shr_in1) >>> in2_imm[SHIFT_IMM_BITS-1:0]);
            end
        endcase
    end
`ifndef VX_CFG_EXT_ZICOND_ENABLE
    `UNUSED_VAR (in2)
`endif

    // AND, OR, XOR, SLL
    always @(*) begin
        case (op)
            2'b00: msc_result = in1 & in2_imm; // AND
            2'b01: msc_result = in1 | in2_imm; // OR
            2'b10: msc_result = in1 ^ in2_imm; // XOR
            2'b11: msc_result = in1 << in2_imm[SHIFT_IMM_BITS-1:0]; // SLL
        endcase
    end

`ifdef VX_CFG_XLEN_64
    wire [32:0] shr_in1_w = {is_signed && in1[31], in1[31:0]};
    wire [31:0] shr_res_w = 32'($signed(shr_in1_w) >>> in2_imm[4:0]);
    assign shr_result_w = `VX_CFG_XLEN'($signed(shr_res_w)); // SRLW, SRAW
    assign msc_result_w = `VX_CFG_XLEN'($signed(in1[31:0] << in2_imm[4:0])); // SLLW
`else
    // W ops exist only on RV64
    assign shr_result_w = '0;
    assign msc_result_w = '0;
`endif

    // WGATHER: the source is the group's lane wg_src_offset, falling back to
    // the last active lane when that lane is masked.
    if (NUM_LANES > 1) begin : g_wgather
        wire [WG_BITS-1:0] nominal = WG_BITS'(wg_src_offset);
        wire [1:0]         offset  = wg_pos - wg_src_offset; // (i - src) mod 4
        // offset 0 = source lane (write-suppressed)
        wire [`VX_CFG_XLEN-1:0] nominal_val = (offset == 2'd1) ? wg_in1[nominal]
                                            : (offset == 2'd2) ? wg_in2[nominal]
                                            : (offset == 2'd3) ? wg_in3[nominal]
                                            : `VX_CFG_XLEN'(0);
        wire [`VX_CFG_XLEN-1:0] last_val    = (offset == 2'd1) ? last_in1
                                            : (offset == 2'd2) ? last_in2
                                            : (offset == 2'd3) ? last_in3
                                            : `VX_CFG_XLEN'(0);
        assign wgather_result = wg_tmask[nominal] ? nominal_val : last_val;
    end else begin : g_wgather_0
        assign wgather_result = in1;
        `UNUSED_VAR ({wg_pos, wg_src_offset, wg_in1, wg_in2, wg_in3, wg_tmask, last_in1, last_in2, last_in3})
    end

endmodule
