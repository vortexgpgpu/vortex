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

module VX_tcu_bhf_bf16mul (
    input wire enable,
    input  wire [15:0] a,           //BF16 input a
    input  wire [15:0] b,           //BF16 input b
    output logic [32:0] y           //FP32 Recoded output y
);

    `UNUSED_VAR(enable);

    //BF16 format constants
    localparam BF16_EXP_WIDTH = 8;
    localparam BF16_SIG_WIDTH = 8;

    //FP32 format constants
    localparam FP32_EXP_WIDTH = 8;
    localparam FP32_SIG_WIDTH = 24;

    //Control and rounding mode
    localparam CONTROL = 1'b1;          //Default (tininess after rounding)
    localparam [2:0] RNE = 3'b000;      //Round Near Even mode

    //Recoded format widths
    localparam BF16_REC_WIDTH = BF16_EXP_WIDTH + BF16_SIG_WIDTH + 1;  //17-bit
    localparam FP32_REC_WIDTH = FP32_EXP_WIDTH + FP32_SIG_WIDTH + 1;  //33-bit

    //Recoded input signals
    wire [BF16_REC_WIDTH-1:0] a_recoded;
    wire [BF16_REC_WIDTH-1:0] b_recoded;

    //Convert BF16 inputs from standard to recoded format
    fNToRecFN #(
        .expWidth(BF16_EXP_WIDTH),
        .sigWidth(BF16_SIG_WIDTH)
    ) conv_a (
        .in(a),
        .out(a_recoded)
    );
    fNToRecFN #(
        .expWidth(BF16_EXP_WIDTH),
        .sigWidth(BF16_SIG_WIDTH)
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
    wire signed [BF16_EXP_WIDTH+1:0] raw_out_sExp;
    wire [(BF16_SIG_WIDTH*2 -1):0] raw_out_sig; //16-bits

    //Mul BF16 inputs to BF16 raw (double width sig)
    mulRecFNToFullRaw #(
        .expWidth(BF16_EXP_WIDTH),
        .sigWidth(BF16_SIG_WIDTH)
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

    wire [FP32_REC_WIDTH-1:0] fp32_recoded;
    wire [4:0] fp32_exception_flags;
    `UNUSED_VAR(fp32_exception_flags)

    //Round raw BF16 result to recoded FP32 format (Required for feeding into addRecFN)
    roundAnyRawFNToRecFN #(
        .inExpWidth(BF16_EXP_WIDTH),
        .inSigWidth(BF16_SIG_WIDTH*2-1),    //15-bits (for 15:0 in_sig instantiation)
        .outExpWidth(FP32_EXP_WIDTH),
        .outSigWidth(FP32_SIG_WIDTH),
        .options(0)
    ) bf16_32rounder (
        .control(CONTROL),
        .invalidExc(raw_invalidExc),
        .infiniteExc(1'b0),
        .in_isNaN(raw_out_isNaN),
        .in_isInf(raw_out_isInf),
        .in_isZero(raw_out_isZero),
        .in_sign(raw_out_sign),
        .in_sExp(raw_out_sExp),
        .in_sig(raw_out_sig),
        .roundingMode(RNE),
        .out(fp32_recoded),
        .exceptionFlags(fp32_exception_flags)
    );

    assign y = fp32_recoded;

    /*
    //Final result exception handling (combine flags from both rounding stages)
    wire [4:0] combined_flags = bf16_exception_flags | fp32_exception_flags;
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
