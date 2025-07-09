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

`define E4M3 //higher precision - forward pass
// `define E5M2 //wider range - backward pass
// (un)comment as required for now
// TODO: add these defs to VX_config.vh and config scripts accordingly

`define FP16_ACC //Accumulate in FP16
// `define FP32_ACC //Accumulate in FP32
// (un)comment as required for now

module VX_tcu_bhf_fp8mul (
    input wire enable,
    input  wire [7:0] a,           //FP8 input a
    input  wire [7:0] b,           //FP8 input b
`ifdef FP16_ACC
    output logic [16:0] y      //FP16 Recoded output y
`elsif FP32_ACC
    output logic [32:0] y      //FP32 Recoded output y
`endif
);

    `UNUSED_VAR(enable);

    //FP8 format constants
    `ifdef E4M3
        localparam FP8_EXP_WIDTH = 4;
        localparam FP8_SIG_WIDTH = 4;
    `elsif E5M2
        localparam FP8_EXP_WIDTH = 5;
        localparam FP8_SIG_WIDTH = 3;
    `endif

    //FP16 format constants
    localparam FP16_EXP_WIDTH = 5;
    localparam FP16_SIG_WIDTH = 11;

    //FP32 format constants
    localparam FP32_EXP_WIDTH = 8;
    localparam FP32_SIG_WIDTH = 24;

    //Control and rounding mode
    localparam CONTROL = 1'b1;          //Default (tininess after rounding) -recommended
    localparam [2:0] RNE = 3'b000;      //Round Near Even mode

    //Recoded format widths
    localparam FP8_REC_WIDTH = FP8_EXP_WIDTH + FP8_SIG_WIDTH + 1;  //9-bit
    localparam FP16_REC_WIDTH = FP16_EXP_WIDTH + FP16_SIG_WIDTH + 1;  //17-bit
    localparam FP32_REC_WIDTH = FP32_EXP_WIDTH + FP32_SIG_WIDTH + 1;  //33-bit

    //Recoded input signals
    wire [FP8_REC_WIDTH-1:0] a_recoded;
    wire [FP8_REC_WIDTH-1:0] b_recoded;

    //Convert FP8 inputs from standard to recoded format
    fNToRecFN #(
        .expWidth(FP8_EXP_WIDTH),
        .sigWidth(FP8_SIG_WIDTH)
    ) conv_a (
        .in(a),
        .out(a_recoded)
    );
    fNToRecFN #(
        .expWidth(FP8_EXP_WIDTH),
        .sigWidth(FP8_SIG_WIDTH)
    ) conv_b (
        .in(b),
        .out(b_recoded)
    );

    //Raw multiplication outputs
    wire raw_invalidExc;
    wire raw_out_isNaN;
    wire raw_out_isInf;
    wire raw_out_isZero;
    wire raw_out_sign;
    wire signed [FP8_EXP_WIDTH+1:0] raw_out_sExp;
    wire [(FP8_SIG_WIDTH*2 -1):0] raw_out_sig; //8-bits(E4M3) or 6-bits(E5M2)

    //Mul FP8 inputs to FP8 full raw (double width sig)
    mulRecFNToFullRaw #(
        .expWidth(FP8_EXP_WIDTH),
        .sigWidth(FP8_SIG_WIDTH)
    ) multiplier (
        .control(CONTROL),
        .a(a_recoded),
        .b(b_recoded),
        .invalidExc(raw_invalidExc),
        .out_isNaN(raw_out_isNaN),
        .out_isInf(raw_out_isInf),
        .out_isZero(raw_out_isZero),
        .out_sign(raw_out_sign),
        .out_sExp(raw_out_sExp),
        .out_sig(raw_out_sig)
    );

    `ifdef FP16_ACC
        wire [FP16_REC_WIDTH-1:0] fp16_recoded;
        wire [4:0] fp16_exception_flags;
        `UNUSED_VAR(fp16_exception_flags)
    `elsif FP32_ACC
        wire [FP32_REC_WIDTH-1:0] fp32_recoded;
        wire [4:0] fp32_exception_flags;
        `UNUSED_VAR(fp32_exception_flags)
    `endif

    //Round raw FP8 result to recoded FP16/FP32 format (Required for feeding into addRecFN)
    roundAnyRawFNToRecFN #(
        .inExpWidth(FP8_EXP_WIDTH),
        .inSigWidth(FP8_SIG_WIDTH*2-1),     //7-bits(E4M3) or 5-bits(E5M2)
        `ifdef FP16_ACC
            .outExpWidth(FP16_EXP_WIDTH),
            .outSigWidth(FP16_SIG_WIDTH),
        `elsif FP32_ACC
            .outExpWidth(FP32_EXP_WIDTH),
            .outSigWidth(FP32_SIG_WIDTH),
        `endif
        .options(0)
    ) fp8_1632rounder (
        .control(CONTROL),
        .invalidExc(raw_invalidExc),
        .infiniteExc(1'b0),
        .in_isNaN(raw_out_isNaN),
        .in_isInf(raw_out_isInf),
        .in_isZero(raw_out_isZero),
        .in_sign(raw_out_sign),
        .in_sExp(raw_out_sExp),
        .in_sig(raw_out_sig),               //[7:0] or [5:0]
        .roundingMode(RNE),
        `ifdef FP16_ACC
            .out(fp16_recoded),
            .exceptionFlags(fp16_exception_flags)
        `ifdef FP32_ACC
            .out(fp32_recoded),
            .exceptionFlags(fp32_exception_flags)
        `endif
    );

    `ifdef FP16_ACC
        assign y = fp16_recoded;
    `endif FP32_ACC
        assign y = fp32_recoded
    `endif

    /*
    //Final result exception handling (combine flags from both rounding stages)
    wire [4:0] combined_flags = FP8_exception_flags | fp32_exception_flags;
    `UNUSED_VAR(combined_flags)
    wire result_is_inf = 1'b0; //combined_flags[3];
    wire result_is_nan = 1'b0; //combined_flags[4] | (|combined_flags[2:0]);

    always_comb begin
        casez({result_is_nan, result_is_inf})
            2'b1?: y = 33'h07FC00000;                                //Canonical FP32 quiet NaN
            2'b01: y = y_wo_exp[32] ? 32'hFF800000 : 32'h7F800000;  //Signed FP32 infinity
            default: y = fp32_recoded;
        endcase
    end
    */

endmodule
