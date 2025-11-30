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

// Modified port of cast module from fpnew Libray
// reference: https://github.com/pulp-platform/fpnew

`include "VX_fpu_define.vh"

`ifdef FPU_DSP

module VX_fcvt_unit import VX_fpu_pkg::*; #(
    parameter LATENCY   = 1,
    parameter INT_WIDTH = 32,
    parameter MAN_BITS  = 23,
    parameter EXP_BITS  = 8,
    parameter OUT_REG   = 0
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [`INST_FRM_BITS-1:0] frm,

    input wire is_itof,
    input wire is_signed,

    input wire [31:0]  dataa,
    output wire [31:0] result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    // Constants
    localparam EXP_BIAS = 2**(EXP_BITS-1)-1;

    // The internal mantissa includes normal bit or an entire integer
    localparam S_MAN_WIDTH = `MAX(1+MAN_BITS, INT_WIDTH);

    // The lower 2p+3 bits of the internal FMA result will be needed for leading-zero detection
    localparam LZC_RESULT_WIDTH = `CLOG2(S_MAN_WIDTH);

    // The internal exponent must be able to represent the smallest denormal input value as signed
    // or the number of bits in an integer
    localparam S_EXP_WIDTH = `MAX(`CLOG2(INT_WIDTH), `MAX(EXP_BITS, `CLOG2(EXP_BIAS + MAN_BITS))) + 1;

    localparam FMT_SHIFT_COMPENSATION = S_MAN_WIDTH - 1 - MAN_BITS;
    localparam NUM_FP_STICKY  = 2 * S_MAN_WIDTH - MAN_BITS - 1;   // removed mantissa, 1. and R
    localparam NUM_INT_STICKY = 2 * S_MAN_WIDTH - INT_WIDTH;  // removed int and R

    // Input processing

    fclass_t fclass;
    VX_fp_classifier #(
        .EXP_BITS (EXP_BITS),
        .MAN_BITS (MAN_BITS)
    ) fp_classifier (
        .exp_i  (dataa[INT_WIDTH-2:MAN_BITS]),
        .man_i  (dataa[MAN_BITS-1:0]),
        .clss_o (fclass)
    );

    wire [S_MAN_WIDTH-1:0] input_mant;
    wire [S_EXP_WIDTH-1:0] input_exp;
    wire                   input_sign;

    wire i2f_sign = dataa[INT_WIDTH-1];
    wire f2i_sign = dataa[INT_WIDTH-1] && is_signed;
    wire [S_MAN_WIDTH-1:0] f2i_mantissa = f2i_sign ? (-dataa) : dataa;
    wire [S_MAN_WIDTH-1:0] i2f_mantissa = S_MAN_WIDTH'({fclass.is_normal, dataa[MAN_BITS-1:0]});
    assign input_exp  = {1'b0, dataa[MAN_BITS +: EXP_BITS]} + S_EXP_WIDTH'({1'b0, fclass.is_subnormal});
    assign input_mant = is_itof ? f2i_mantissa : i2f_mantissa;
    assign input_sign = is_itof ? f2i_sign : i2f_sign;

    // Pipeline stage0

    wire                   is_itof_s0;
    wire                   is_signed_s0;
    wire [2:0]             rnd_mode_s0;
    fclass_t               fclass_s0;
    wire                   input_sign_s0;
    wire [S_EXP_WIDTH-1:0] fmt_exponent_s0;
    wire [S_MAN_WIDTH-1:0] encoded_mant_s0;

    VX_pipe_register #(
        .DATAW (1 + `INST_FRM_BITS + 1 + $bits(fclass_t) + 1 + S_EXP_WIDTH + S_MAN_WIDTH),
        .DEPTH (LATENCY > 1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof,    is_signed,    frm,         fclass,    input_sign,    input_exp,       input_mant}),
        .data_out ({is_itof_s0, is_signed_s0, rnd_mode_s0, fclass_s0, input_sign_s0, fmt_exponent_s0, encoded_mant_s0})
    );

    // Normalization

    wire [LZC_RESULT_WIDTH-1:0] renorm_shamt_s0; // renormalization shift amount
    wire mant_is_nonzero_s0;

    VX_lzc #(
        .N (S_MAN_WIDTH)
    ) lzc (
        .data_in   (encoded_mant_s0),
        .data_out  (renorm_shamt_s0),
        .valid_out (mant_is_nonzero_s0)
    );

    wire mant_is_zero_s0 = ~mant_is_nonzero_s0;

    wire [S_MAN_WIDTH-1:0] input_mant_n_s0;    // normalized input mantissa
    wire [S_EXP_WIDTH-1:0] input_exp_n_s0;     // unbiased true exponent

    // Realign input mantissa, append zeroes if destination is wider
    assign input_mant_n_s0 = encoded_mant_s0 << renorm_shamt_s0;

    // Unbias exponent and compensate for shift
    wire [S_EXP_WIDTH-1:0] i2f_input_exp_s0 = fmt_exponent_s0 + S_EXP_WIDTH'(FMT_SHIFT_COMPENSATION - EXP_BIAS) - S_EXP_WIDTH'({1'b0, renorm_shamt_s0});
    wire [S_EXP_WIDTH-1:0] f2i_input_exp_s0 = S_EXP_WIDTH'(S_MAN_WIDTH-1) - S_EXP_WIDTH'({1'b0, renorm_shamt_s0});
    assign input_exp_n_s0 = is_itof_s0 ? f2i_input_exp_s0 : i2f_input_exp_s0;

    // Pipeline stage1

    wire                   is_itof_s1;
    wire                   is_signed_s1;
    wire [2:0]             rnd_mode_s1;
    fclass_t               fclass_s1;
    wire                   input_sign_s1;
    wire                   mant_is_zero_s1;
    wire [S_MAN_WIDTH-1:0] input_mant_s1;
    wire [S_EXP_WIDTH-1:0] input_exp_s1;

    VX_pipe_register #(
        .DATAW (1 + `INST_FRM_BITS + 1 + $bits(fclass_t) + 1 + 1 + S_MAN_WIDTH + S_EXP_WIDTH),
        .DEPTH (LATENCY > 2)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s0, is_signed_s0, rnd_mode_s0, fclass_s0, input_sign_s0, mant_is_zero_s0, input_mant_n_s0, input_exp_n_s0}),
        .data_out ({is_itof_s1, is_signed_s1, rnd_mode_s1, fclass_s1, input_sign_s1, mant_is_zero_s1, input_mant_s1,   input_exp_s1})
    );

    // Perform adjustments to mantissa and exponent

    wire [S_EXP_WIDTH-1:0] denorm_shamt = S_EXP_WIDTH'(INT_WIDTH-1) - input_exp_s1;
    wire overflow = ($signed(denorm_shamt) <= -$signed(S_EXP_WIDTH'(!is_signed_s1)));
    wire underflow = ($signed(input_exp_s1) < S_EXP_WIDTH'($signed(-1)));
    reg [S_EXP_WIDTH-1:0] denorm_shamt_q;
    always @(*) begin
        if (overflow) begin
            denorm_shamt_q = '0;
        end else if (underflow) begin
            denorm_shamt_q = INT_WIDTH+1;
        end else begin
            denorm_shamt_q = denorm_shamt;
        end
    end
    wire [2*S_MAN_WIDTH:0] destination_mant_s1 = is_itof_s1 ? {input_mant_s1, 33'b0} : ({input_mant_s1, 33'b0} >> denorm_shamt_q);
    wire [EXP_BITS-1:0] final_exp_s1 = input_exp_s1[EXP_BITS-1:0] + EXP_BITS'(EXP_BIAS);
    wire of_before_round_s1 = overflow;

    // Pipeline stage2

    wire                   is_itof_s2;
    wire                   is_signed_s2;
    wire [2:0]             rnd_mode_s2;
    fclass_t               fclass_s2;
    wire                   mant_is_zero_s2;
    wire                   input_sign_s2;
    wire [2*S_MAN_WIDTH:0] destination_mant_s2;
    wire [EXP_BITS-1:0]    final_exp_s2;
    wire                   of_before_round_s2;

    VX_pipe_register #(
        .DATAW (1 + 1 + `INST_FRM_BITS + $bits(fclass_t) + 1 + 1 + (2*S_MAN_WIDTH+1) + EXP_BITS + 1),
        .DEPTH (LATENCY > 0)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s1, is_signed_s1, rnd_mode_s1, fclass_s1, mant_is_zero_s1, input_sign_s1, destination_mant_s1, final_exp_s1, of_before_round_s1}),
        .data_out ({is_itof_s2, is_signed_s2, rnd_mode_s2, fclass_s2, mant_is_zero_s2, input_sign_s2, destination_mant_s2, final_exp_s2, of_before_round_s2})
    );

    // Rouding and classification

    wire [MAN_BITS-1:0]  final_mant_s2;  // mantissa after adjustments
    wire [INT_WIDTH-1:0] final_int_s2;   // integer shifted in position
    wire [1:0]           f2i_round_sticky_bits_s2, i2f_round_sticky_bits_s2;

    // Extract final mantissa and round bit, discard the normal bit (for FP)
    assign {final_mant_s2, i2f_round_sticky_bits_s2[1]} = destination_mant_s2[2*S_MAN_WIDTH-1 : 2*S_MAN_WIDTH-1 - (MAN_BITS+1) + 1];
    assign {final_int_s2, f2i_round_sticky_bits_s2[1]} = destination_mant_s2[2*S_MAN_WIDTH   : 2*S_MAN_WIDTH   - (INT_WIDTH+1) + 1];

    // Collapse sticky bits
    assign i2f_round_sticky_bits_s2[0] = (| destination_mant_s2[NUM_FP_STICKY-1:0]);
    assign f2i_round_sticky_bits_s2[0] = (| destination_mant_s2[NUM_INT_STICKY-1:0]);
    wire i2f_round_has_sticky_s2 = (| i2f_round_sticky_bits_s2);
    wire f2i_round_has_sticky_s2 = (| f2i_round_sticky_bits_s2);

    // select RS bits for destination operation
    wire [1:0] round_sticky_bits_s2 = is_itof_s2 ? i2f_round_sticky_bits_s2 : f2i_round_sticky_bits_s2;

    // Pack exponent and mantissa into proper rounding form
    wire [INT_WIDTH-1:0] fmt_pre_round_abs_s2 = {1'b0, final_exp_s2, final_mant_s2[MAN_BITS-1:0]};

    // Select output with destination format and operation
    wire [INT_WIDTH-1:0] pre_round_abs_s2 = is_itof_s2 ? fmt_pre_round_abs_s2 : final_int_s2;

    wire [INT_WIDTH-1:0] rounded_abs_s2;
    wire rounded_sign_s2;

    // Perform the rounding
    VX_fp_rounding #(
        .DAT_WIDTH (32)
    ) fp_rounding (
        .abs_value_i (pre_round_abs_s2),
        .sign_i      (input_sign_s2),
        .round_sticky_bits_i (round_sticky_bits_s2),
        .rnd_mode_i  (rnd_mode_s2),
        .effective_subtraction_i (1'b0),
        .abs_rounded_o (rounded_abs_s2),
        .sign_o      (rounded_sign_s2),
        `UNUSED_PIN  (exact_zero_o)
    );

    // Pipeline stage3

    wire                 is_itof_s3;
    wire                 is_signed_s3;
    fclass_t             fclass_s3;
    wire                 mant_is_zero_s3;
    wire                 input_sign_s3;
    wire                 rounded_sign_s3;
    wire [INT_WIDTH-1:0] rounded_abs_s3;
    wire                 of_before_round_s3;
    wire                 f2i_round_has_sticky_s3;
    wire                 i2f_round_has_sticky_s3;

    `UNUSED_VAR (fclass_s3)

    VX_pipe_register #(
        .DATAW (1 + 1 + $bits(fclass_t) + 1 + 1 + 32 + 1 + 1 + 1 + 1),
        .DEPTH (LATENCY > 3)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s2, is_signed_s2, fclass_s2, mant_is_zero_s2, input_sign_s2, rounded_abs_s2, rounded_sign_s2, of_before_round_s2, f2i_round_has_sticky_s2, i2f_round_has_sticky_s2}),
        .data_out ({is_itof_s3, is_signed_s3, fclass_s3, mant_is_zero_s3, input_sign_s3, rounded_abs_s3, rounded_sign_s3, of_before_round_s3, f2i_round_has_sticky_s3, i2f_round_has_sticky_s3})
    );

    // Assemble regular result, nan box short ones. Int zeroes need to be detected
    wire [INT_WIDTH-1:0] fmt_result_s3 = mant_is_zero_s3 ? 0 : {rounded_sign_s3, rounded_abs_s3[EXP_BITS+MAN_BITS-1:0]};

    // Negative integer result needs to be brought into two's complement
    wire [INT_WIDTH-1:0] rounded_int_res_s3 = rounded_sign_s3 ? (-rounded_abs_s3) : rounded_abs_s3;
    wire rounded_int_res_zero_s3 = (rounded_int_res_s3 == 0);

    // F2I Special case handling

    // Assemble result according to destination format
    reg [INT_WIDTH-1:0] f2i_special_result_s3;
    always @(*) begin
        if (input_sign_s3 && !fclass_s3.is_nan) begin
            f2i_special_result_s3[INT_WIDTH-2:0] = '0;            // alone yields 2**(31)-1
            f2i_special_result_s3[INT_WIDTH-1]   = is_signed_s3;  // for unsigned casts yields 2**31
        end else begin
            f2i_special_result_s3[INT_WIDTH-2:0] = 2**(INT_WIDTH-1) - 1;   // alone yields 2**(31)-1
            f2i_special_result_s3[INT_WIDTH-1]   = ~is_signed_s3;   // for unsigned casts yields 2**31
        end
    end

    // Detect special case from source format (inf, nan, overflow, nan-boxing or negative unsigned)
    wire f2i_result_is_special_s3 = fclass_s3.is_nan
                                  | fclass_s3.is_inf
                                  | of_before_round_s3
                                  | (input_sign_s3 & ~is_signed_s3 & ~rounded_int_res_zero_s3);

    fflags_t f2i_special_status_s3;
    fflags_t i2f_status_s3, f2i_status_s3;
    fflags_t tmp_fflags_s3;

    // All integer special cases are invalid
    assign f2i_special_status_s3 = {1'b1, 4'h0};

    // Result selection and output

    assign i2f_status_s3 = {4'h0, i2f_round_has_sticky_s3};
    assign f2i_status_s3 = f2i_result_is_special_s3 ? f2i_special_status_s3 : {4'h0, f2i_round_has_sticky_s3};

    wire [INT_WIDTH-1:0] i2f_result_s3 = fmt_result_s3;
    wire [INT_WIDTH-1:0] f2i_result_s3 = f2i_result_is_special_s3 ? f2i_special_result_s3 : rounded_int_res_s3;

    wire [INT_WIDTH-1:0] tmp_result_s3 = is_itof_s3 ? i2f_result_s3 : f2i_result_s3;
    assign tmp_fflags_s3 = is_itof_s3 ? i2f_status_s3 : f2i_status_s3;

    VX_pipe_register #(
        .DATAW (32 + `FP_FLAGS_BITS),
        .DEPTH (OUT_REG)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({tmp_result_s3, tmp_fflags_s3}),
        .data_out ({result,        fflags})
    );

endmodule
`endif
