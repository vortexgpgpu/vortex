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

`include "VX_fpu_define.vh"

`ifdef FPU_TYPE_DSP

// F32-only float conversion unit (I2F and F2I).
// F64 (double) support removed; XLEN-wide integer (I32/I64) support kept.
// LATENCY: use 5 for XLEN=32, 6 for XLEN=64 (separates S0 negate from S1 LZC).
module VX_fcvt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*;
#(
    parameter LATENCY   = 5,
    parameter XLEN      = `XLEN,
    parameter OUT_REG   = 0
) (
    input wire clk,
    input wire reset,

    input wire enable,

    input wire [INST_FRM_BITS-1:0] frm,

    input wire is_itof,    // Integer → F32
    input wire is_ftoi,    // F32 → Integer
    input wire is_signed,  // signed integer conversion
    input wire is_dst_64,  // F2I: 1=I64 target, 0=I32 target
    input wire is_src_64,  // I2F: 1=I64 source, 0=I32 source

    input wire [XLEN-1:0]  dataa,
    output wire [XLEN-1:0] result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    // F32-only constants
    localparam F32_EXP  = 8;
    localparam F32_MAN  = 23;
    localparam F32_BIAS = 127;

    localparam MAX_MAN = F32_MAN;
    localparam MAX_EXP = F32_EXP;

    // Internal widths driven by XLEN (for I64 support) and F32 mantissa
    localparam S_MAN_WIDTH      = `MAX(1 + MAX_MAN, XLEN);
    localparam S_EXP_WIDTH      = `MAX(`CLOG2(XLEN), MAX_EXP + 2) + 2;
    localparam LZC_RESULT_WIDTH = `CLOG2(S_MAN_WIDTH);

    // ======================================================================
    // Stage 0: Unpack & Classify
    // ======================================================================

    wire [63:0] safe_dataa = 64'(dataa);

    // F32-only extraction (no F64 source path)
    wire [MAX_EXP-1:0] input_fp_exp = safe_dataa[F32_MAN +: F32_EXP];
    wire [MAX_MAN-1:0] input_fp_man = safe_dataa[F32_MAN-1:0];
    wire               input_fp_sgn = safe_dataa[31];

    fclass_t fclass;
    VX_fp_classifier #(
        .EXP_BITS (MAX_EXP),
        .MAN_BITS (MAX_MAN)
    ) fp_classifier (
        .exp_i  (input_fp_exp),
        .man_i  (input_fp_man),
        .clss_o (fclass)
    );

    wire i_sign = safe_dataa[is_src_64 ? (XLEN-1) : 31] && is_signed;
    wire [XLEN-1:0] i_mag_raw = i_sign ? (-dataa) : dataa;
    wire [XLEN-1:0] i_mag = is_src_64 ? i_mag_raw : XLEN'(i_mag_raw[31:0]);

    logic [S_MAN_WIDTH-1:0]        unpacked_mant_s0;
    logic signed [S_EXP_WIDTH-1:0] unpacked_exp_s0;
    wire input_sign_s0 = is_itof ? i_sign : input_fp_sgn;

    always_comb begin
        if (is_itof) begin
            unpacked_mant_s0 = S_MAN_WIDTH'(i_mag);
            unpacked_exp_s0  = S_EXP_WIDTH'(XLEN - 1);
        end else begin
            unpacked_mant_s0 = S_MAN_WIDTH'({fclass.is_normal, input_fp_man});
            unpacked_exp_s0  = S_EXP_WIDTH'(input_fp_exp) - S_EXP_WIDTH'(F32_BIAS)
                             + S_EXP_WIDTH'({1'b0, fclass.is_subnormal});
        end
    end

    // Stage 0 -> Stage 1
    wire is_itof_s1, is_ftoi_s1, is_signed_s1, is_dst_64_s1, input_sign_s1;
    wire [INST_FRM_BITS-1:0] frm_s1;
    fclass_t fclass_s1;
    wire [S_MAN_WIDTH-1:0]        unpacked_mant_s1;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s1;

    VX_pipe_register #(
        .DATAW (5 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH),
        .DEPTH (LATENCY > 5)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof,    is_ftoi,    is_signed,    is_dst_64,    input_sign_s0, frm,   fclass,    unpacked_mant_s0, unpacked_exp_s0}),
        .data_out ({is_itof_s1, is_ftoi_s1, is_signed_s1, is_dst_64_s1, input_sign_s1, frm_s1, fclass_s1, unpacked_mant_s1, unpacked_exp_s1})
    );

    // ======================================================================
    // Stage 1: Leading Zero Count (LZC)
    // ======================================================================

    wire [LZC_RESULT_WIDTH-1:0] renorm_shamt_s1;
    wire mant_is_nonzero_s1;

    VX_lzc #(
        .N (S_MAN_WIDTH)
    ) lzc (
        .data_in   (unpacked_mant_s1),
        .data_out  (renorm_shamt_s1),
        .valid_out (mant_is_nonzero_s1)
    );

    wire mant_is_zero_s1 = ~mant_is_nonzero_s1;

    // Stage 1 -> Stage 2
    wire is_itof_s2, is_ftoi_s2, is_signed_s2, is_dst_64_s2, input_sign_s2, mant_is_zero_s2;
    wire [INST_FRM_BITS-1:0] frm_s2;
    fclass_t fclass_s2;
    wire [S_MAN_WIDTH-1:0]        unpacked_mant_s2;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s2;
    wire [LZC_RESULT_WIDTH-1:0]   renorm_shamt_s2;

    VX_pipe_register #(
        .DATAW (6 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH + LZC_RESULT_WIDTH),
        .DEPTH (LATENCY > 4)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s1, is_ftoi_s1, is_signed_s1, is_dst_64_s1, input_sign_s1, mant_is_zero_s1, frm_s1, fclass_s1, unpacked_mant_s1, unpacked_exp_s1, renorm_shamt_s1}),
        .data_out ({is_itof_s2, is_ftoi_s2, is_signed_s2, is_dst_64_s2, input_sign_s2, mant_is_zero_s2, frm_s2, fclass_s2, unpacked_mant_s2, unpacked_exp_s2, renorm_shamt_s2})
    );

    // ======================================================================
    // Stage 2: Normalize (Left Shift)
    // ======================================================================

    // F32 mantissa {normal_bit, man[22:0]} occupies bits [MAX_MAN:0] of the S_MAN_WIDTH field.
    // LZC counts leading zeros including the padding: subtract them back for FP inputs.
    localparam FP_MANT_OFFSET = S_MAN_WIDTH - 1 - MAX_MAN;

    wire [S_MAN_WIDTH-1:0]        norm_mant_s2 = unpacked_mant_s2 << renorm_shamt_s2;
    wire signed [S_EXP_WIDTH-1:0] norm_exp_s2  = unpacked_exp_s2 - S_EXP_WIDTH'({1'b0, renorm_shamt_s2})
        + (is_itof_s2 ? S_EXP_WIDTH'(0) : S_EXP_WIDTH'(FP_MANT_OFFSET));

    // Stage 2 -> Stage 3
    wire is_itof_s3, is_ftoi_s3, is_signed_s3, is_dst_64_s3, input_sign_s3, mant_is_zero_s3;
    wire [INST_FRM_BITS-1:0] frm_s3;
    fclass_t fclass_s3;
    wire [S_MAN_WIDTH-1:0]        norm_mant_s3;
    wire signed [S_EXP_WIDTH-1:0] norm_exp_s3;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s3; // pre-renorm exp, needed for f2i_shamt

    VX_pipe_register #(
        .DATAW (6 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + 2*S_EXP_WIDTH),
        .DEPTH (LATENCY > 3)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s2, is_ftoi_s2, is_signed_s2, is_dst_64_s2, input_sign_s2, mant_is_zero_s2, frm_s2, fclass_s2, norm_mant_s2, norm_exp_s2,  unpacked_exp_s2}),
        .data_out ({is_itof_s3, is_ftoi_s3, is_signed_s3, is_dst_64_s3, input_sign_s3, mant_is_zero_s3, frm_s3, fclass_s3, norm_mant_s3, norm_exp_s3,  unpacked_exp_s3})
    );

    // ======================================================================
    // Stage 3: Calculate Alignment Shift
    // ======================================================================

    // F2I: use pre-renorm exponent to avoid double-shifting
    wire signed [S_EXP_WIDTH-1:0] f2i_shamt_s3 = S_EXP_WIDTH'(S_MAN_WIDTH - 1) - unpacked_exp_s3;

    logic [S_EXP_WIDTH-1:0] align_shamt_s3;
    always_comb begin
        if (is_itof_s3) begin
            align_shamt_s3 = '0;
        end else begin // F2I (no F2F: only F2I reaches here)
            align_shamt_s3 = (f2i_shamt_s3 > S_EXP_WIDTH'(S_MAN_WIDTH + 1)) ? S_EXP_WIDTH'(S_MAN_WIDTH + 1) :
                             (f2i_shamt_s3 < 0) ? '0 : f2i_shamt_s3;
        end
    end

    // F32-only: always bias with F32_BIAS
    wire [MAX_EXP-1:0] final_exp_s3 = MAX_EXP'(norm_exp_s3 + S_EXP_WIDTH'(F32_BIAS));

    // Stage 3 -> Stage 4
    wire is_itof_s4, is_ftoi_s4, is_signed_s4, is_dst_64_s4, input_sign_s4, mant_is_zero_s4;
    wire [INST_FRM_BITS-1:0] frm_s4;
    fclass_t fclass_s4;
    wire [S_MAN_WIDTH-1:0]  norm_mant_s4;
    wire [S_EXP_WIDTH-1:0]  align_shamt_s4;
    wire [MAX_EXP-1:0]      final_exp_s4;

    VX_pipe_register #(
        .DATAW (6 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH + MAX_EXP),
        .DEPTH (LATENCY > 2)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s3, is_ftoi_s3, is_signed_s3, is_dst_64_s3, input_sign_s3, mant_is_zero_s3, frm_s3, fclass_s3, norm_mant_s3, align_shamt_s3, final_exp_s3}),
        .data_out ({is_itof_s4, is_ftoi_s4, is_signed_s4, is_dst_64_s4, input_sign_s4, mant_is_zero_s4, frm_s4, fclass_s4, norm_mant_s4, align_shamt_s4, final_exp_s4})
    );

    // ======================================================================
    // Stage 4: Align & Extract Sticky (Right Shift)
    // ======================================================================

    wire [2*S_MAN_WIDTH:0] aligned_mant_full_s4 = {norm_mant_s4, {(S_MAN_WIDTH+1){1'b0}}} >> align_shamt_s4;

    wire [S_MAN_WIDTH-1:0] aligned_mant_s4 = aligned_mant_full_s4[2*S_MAN_WIDTH : S_MAN_WIDTH+1];
    wire                   guard_bit_s4    = aligned_mant_full_s4[S_MAN_WIDTH];
    wire                   round_bit_s4    = aligned_mant_full_s4[S_MAN_WIDTH-1];
    wire                   sticky_bit_s4   = |aligned_mant_full_s4[S_MAN_WIDTH-2:0];

    // F32-only: dst_man_width is always F32_MAN=23
    wire [S_MAN_WIDTH-1:0] fp_trunc_s4      = aligned_mant_s4 >> (S_MAN_WIDTH - 1 - F32_MAN);
    wire [S_MAN_WIDTH-1:0] trunc_mask_s4    = (S_MAN_WIDTH'(1) << (S_MAN_WIDTH - 1 - F32_MAN)) - 1;
    wire                   sticky_red_s4    = |(aligned_mant_s4 & trunc_mask_s4) | guard_bit_s4 | round_bit_s4 | sticky_bit_s4;
    wire                   fp_guard_s4      = fp_trunc_s4[0];

    logic [1:0]            round_sticky_bits_s4;
    logic [S_MAN_WIDTH-1:0] pre_round_abs_s4;

    always_comb begin
        if (is_ftoi_s4) begin // F2I
            round_sticky_bits_s4 = {round_bit_s4, sticky_bit_s4};
            pre_round_abs_s4     = aligned_mant_s4;
        end else begin // I2F
            round_sticky_bits_s4 = {fp_guard_s4, sticky_red_s4};
            pre_round_abs_s4     = fp_trunc_s4;
        end
    end

    // Stage 4 -> Stage 5
    wire is_itof_s5, is_ftoi_s5, is_signed_s5, is_dst_64_s5, input_sign_s5, mant_is_zero_s5;
    wire [INST_FRM_BITS-1:0] frm_s5;
    fclass_t fclass_s5;
    wire [1:0]            round_sticky_bits_s5;
    wire [S_MAN_WIDTH-1:0] pre_round_abs_s5;
    wire [MAX_EXP-1:0]    final_exp_s5;

    VX_pipe_register #(
        .DATAW (6 + INST_FRM_BITS + $bits(fclass_t) + 2 + S_MAN_WIDTH + MAX_EXP),
        .DEPTH (LATENCY > 1)
    ) pipe_reg5 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({is_itof_s4, is_ftoi_s4, is_signed_s4, is_dst_64_s4, input_sign_s4, mant_is_zero_s4, frm_s4, fclass_s4, round_sticky_bits_s4, pre_round_abs_s4, final_exp_s4}),
        .data_out ({is_itof_s5, is_ftoi_s5, is_signed_s5, is_dst_64_s5, input_sign_s5, mant_is_zero_s5, frm_s5, fclass_s5, round_sticky_bits_s5, pre_round_abs_s5, final_exp_s5})
    );

    // ======================================================================
    // Stage 5: Rounding & Pack
    // ======================================================================

    wire [S_MAN_WIDTH-1:0] rounded_abs_s5;
    wire rounded_sign_s5;

    VX_fp_rounding #(
        .DAT_WIDTH (S_MAN_WIDTH)
    ) fp_rounding (
        .abs_value_i         (pre_round_abs_s5),
        .sign_i              (input_sign_s5),
        .round_sticky_bits_i (round_sticky_bits_s5),
        .rnd_mode_i          (frm_s5),
        .effective_subtraction_i (1'b0),
        .abs_rounded_o       (rounded_abs_s5),
        .sign_o              (rounded_sign_s5),
        `UNUSED_PIN          (exact_zero_o)
    );

    wire [63:0] safe_rounded_abs = 64'(rounded_abs_s5);
    wire [15:0] final_exp_16     = 16'(final_exp_s5);
    wire [15:0] safe_dst_exp     = mant_is_zero_s5 ? 16'h0 : final_exp_16;

    // Integer result (F2I)
    wire [63:0] abs_xlen_64  = 64'(rounded_abs_s5[XLEN-1:0]);
    wire [63:0] safe_int_res = rounded_sign_s5 ? (-abs_xlen_64) : abs_xlen_64;
    wire [31:0] int_32_res   = safe_int_res[31:0];
    wire [63:0] int_64_res   = safe_int_res;

    // F32-only packing (no F64 result)
    wire f32_man_ovf         = rounded_abs_s5[F32_MAN + 1];
    wire [F32_EXP-1:0] fp32_exp = safe_dst_exp[F32_EXP-1:0] + F32_EXP'(f32_man_ovf);
    wire [F32_MAN-1:0] fp32_man = f32_man_ovf ? '0 : safe_rounded_abs[F32_MAN-1:0];
    wire [31:0] fp_32_res    = {rounded_sign_s5, fp32_exp, fp32_man};

    // F2I saturation values for NaN/Inf inputs
    wire use_neg_sat_s5 = fclass_s5.is_inf && input_sign_s5;
    wire [31:0] nan_inf_32 = use_neg_sat_s5 ? (is_signed_s5 ? 32'h80000000 : 32'h00000000)
                                             : (is_signed_s5 ? 32'h7FFFFFFF : 32'hFFFFFFFF);
    wire [63:0] nan_inf_64 = use_neg_sat_s5 ? (is_signed_s5 ? 64'h8000000000000000 : 64'h0000000000000000)
                                             : (is_signed_s5 ? 64'h7FFFFFFFFFFFFFFF : 64'hFFFFFFFFFFFFFFFF);

    // F2I overflow detection (I32 target)
    wire f2i_s32_pos_ovf = is_signed_s5 && !is_dst_64_s5 && !input_sign_s5 && (|rounded_abs_s5[S_MAN_WIDTH-1:31]);
    wire f2i_s32_neg_ovf = is_signed_s5 && !is_dst_64_s5 &&  input_sign_s5 && (rounded_abs_s5 > S_MAN_WIDTH'(32'h80000000));
    wire f2i_u32_neg_ovf = !is_signed_s5 && !is_dst_64_s5 && rounded_sign_s5 && (|rounded_abs_s5);

    // F2I overflow detection (I64 target, XLEN=64 only)
    wire f2i_s64_pos_ovf, f2i_s64_neg_ovf, f2i_u64_neg_ovf;
    if (XLEN == 64) begin : g_f2i_64ovf
        assign f2i_s64_pos_ovf = is_signed_s5 && is_dst_64_s5 && !input_sign_s5 && (|rounded_abs_s5[S_MAN_WIDTH-1:63]);
        assign f2i_s64_neg_ovf = is_signed_s5 && is_dst_64_s5 &&  input_sign_s5 && (rounded_abs_s5 > S_MAN_WIDTH'(64'h8000000000000000));
        assign f2i_u64_neg_ovf = !is_signed_s5 && is_dst_64_s5 && rounded_sign_s5 && (|rounded_abs_s5);
    end else begin : g_f2i_no64ovf
        assign f2i_s64_pos_ovf = 1'b0;
        assign f2i_s64_neg_ovf = 1'b0;
        assign f2i_u64_neg_ovf = 1'b0;
    end

    logic [63:0] res_val_64;
    logic [31:0] res_val_32;
    fflags_t final_fflags_s5;

    always_comb begin
        final_fflags_s5 = '0;
        res_val_64 = '0;
        res_val_32 = '0;

        if (is_ftoi_s5) begin
            // F2I: F32 → I32/I64
            if (fclass_s5.is_nan || fclass_s5.is_inf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = is_dst_64_s5 ? nan_inf_64 : {32'h00000000, nan_inf_32};
                res_val_32 = nan_inf_32;
            end else if (f2i_s32_pos_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = {32'h00000000, 32'h7FFFFFFF};
                res_val_32 = 32'h7FFFFFFF;
            end else if (f2i_s32_neg_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = {{32{1'b1}}, 32'h80000000};
                res_val_32 = 32'h80000000;
            end else if (f2i_u32_neg_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = '0;
                res_val_32 = '0;
            end else if (f2i_s64_pos_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = 64'h7FFFFFFFFFFFFFFF;
                res_val_32 = 32'h7FFFFFFF;
            end else if (f2i_s64_neg_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = 64'h8000000000000000;
                res_val_32 = 32'h80000000;
            end else if (f2i_u64_neg_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = '0;
                res_val_32 = '0;
            end else begin
                res_val_64 = is_dst_64_s5 ? int_64_res : {{32{int_32_res[31]}}, int_32_res};
                res_val_32 = int_32_res;
                final_fflags_s5.NX = (|round_sticky_bits_s5);
            end
        end else begin
            // I2F: Integer → F32 (NaN-box result in 64-bit register)
            res_val_64 = {32'hFFFFFFFF, fp_32_res};
            res_val_32 = fp_32_res;
            final_fflags_s5.NX = (|round_sticky_bits_s5);
        end
    end

    `UNUSED_VAR ({is_itof_s5, fclass_s5, safe_rounded_abs, safe_dst_exp, res_val_64, res_val_32})

    wire [XLEN-1:0] final_result_s5;
    if (XLEN == 64) begin : g_out_x64
        assign final_result_s5 = res_val_64;
    end else begin : g_out_x32
        assign final_result_s5 = res_val_32;
    end

    // Stage 5 -> Output Register
    VX_pipe_register #(
        .DATAW (XLEN + `FP_FLAGS_BITS),
        .DEPTH (OUT_REG)
    ) pipe_reg_out (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({final_result_s5, final_fflags_s5}),
        .data_out ({result,          fflags})
    );

endmodule

`endif
