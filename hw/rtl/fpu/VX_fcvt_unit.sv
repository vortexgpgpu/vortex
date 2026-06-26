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

// Merged conversion unit (I2F, F2I, F2F). XLEN-wide integers (I32/I64).
// FLEN=32 handles F32 only; FLEN=64 adds F64 and F2F (FCVT.S.D / FCVT.D.S),
// selected at runtime by src_fmt/dst_fmt (0=F32, 1=F64). The canonical
// datapath is sized to the widest enabled float format (SUPER_*); narrower
// formats occupy the low bits. FLEN=32 elaborates to the original F32 logic.
// LATENCY: 5 for XLEN=32, 6 for XLEN=64 (separates S0 negate from S1 LZC).
module VX_fcvt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*;
#(
    parameter LATENCY   = 5,
    parameter FLEN      = 32,
    parameter OUT_REG   = 0,
    // 1: full IEEE subnormal support. 0: flush-to-zero (DAZ subnormal float
    //    sources to signed zero) for area. Use 0 for relaxed paths.
    parameter SNORM_ENABLE = 1,
    // 1: produce fflags (NV/OF/UF/NX). 0: tie fflags to 0 (area).
    parameter EXCEPT_ENABLE  = 1
) (
    input wire clk,
    input wire reset,

    input wire enable,
    input wire mask,

    input wire [INST_FRM_BITS-1:0] frm,

    input wire is_itof,    // Integer → Float
    input wire is_ftoi,    // Float → Integer
    input wire is_f2f,     // Float → Float (FLEN=64 only)
    input wire is_signed,  // signed integer conversion
    input wire is_int64,   // integer operand width: 1=I64, 0=I32
    input wire src_fmt,    // source float format: 1=F64, 0=F32
    input wire dst_fmt,    // dest   float format: 1=F64, 0=F32

    input wire [`VX_CFG_XLEN-1:0]  dataa,
    output wire [`VX_CFG_XLEN-1:0] result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam F32_EXP  = 8,  F32_MAN = 23, F32_BIAS = 127;
    localparam F64_EXP  = 11, F64_MAN = 52, F64_BIAS = 1023;
    localparam HAS_D    = (FLEN >= 64);

    // Widest enabled float format drives the canonical internal datapath.
    localparam SUPER_MAN = HAS_D ? F64_MAN : F32_MAN;
    localparam SUPER_EXP = HAS_D ? F64_EXP : F32_EXP;

    // Internal widths driven by XLEN (for I64 support) and the super mantissa.
    localparam S_MAN_WIDTH      = `MAX(1 + SUPER_MAN, `VX_CFG_XLEN);
    localparam S_EXP_WIDTH      = `MAX(`CLOG2(`VX_CFG_XLEN), SUPER_EXP + 2) + 2;
    localparam LZC_RESULT_WIDTH = `CLOG2(S_MAN_WIDTH);

    `UNUSED_VAR (src_fmt)
    `UNUSED_VAR (dst_fmt)
    `UNUSED_VAR (is_f2f)
    wire src_is_d = HAS_D & src_fmt;
    wire dst_is_d = HAS_D & dst_fmt;
    wire f2f      = HAS_D & is_f2f;

    reg [LATENCY-1:0] mask_pipe;
    always @(posedge clk) begin
        if (reset) begin
            mask_pipe <= '0;
        end else if (enable) begin
            mask_pipe <= {mask_pipe[LATENCY-2:0], mask};
        end
    end
    localparam STG2_CYC = LATENCY - 5;
    wire stg2_mask = (STG2_CYC == 0) ? mask : mask_pipe[STG2_CYC-1];

    // ======================================================================
    // Stage 0: Unpack & Classify
    // ======================================================================

    wire [63:0] safe_dataa = 64'(dataa);

    // Float source field extraction (per source format)
    wire               input_fp_sgn = src_is_d ? safe_dataa[63] : safe_dataa[31];

    fclass_t fclass32;
    VX_fp_classifier #(.EXP_BITS(F32_EXP), .MAN_BITS(F32_MAN)) fp_classifier32 (
        .exp_i  (safe_dataa[F32_MAN +: F32_EXP]),
        .man_i  (safe_dataa[F32_MAN-1:0]),
        .clss_o (fclass32)
    );

    fclass_t fclass;
    if (HAS_D) begin : g_classify_d
        fclass_t fclass64;
        VX_fp_classifier #(.EXP_BITS(F64_EXP), .MAN_BITS(F64_MAN)) fp_classifier64 (
            .exp_i  (safe_dataa[F64_MAN +: F64_EXP]),
            .man_i  (safe_dataa[F64_MAN-1:0]),
            .clss_o (fclass64)
        );
        assign fclass = src_is_d ? fclass64 : fclass32;
    end else begin : g_classify_s
        assign fclass = fclass32;
    end

    // Source significand {hidden, mantissa} placed in the low bits, and the
    // true source exponent (biased − bias). Both reduce to F32 when !HAS_D.
    // DAZ: when subnormals are disabled, a subnormal float source is treated as
    // signed zero (mantissa forced 0 -> downstream zero path).
    wire src_daz = (SNORM_ENABLE == 0) & fclass.is_subnormal;
    wire [S_MAN_WIDTH-1:0] fp_unpacked_mant = src_daz ? '0
        : src_is_d ? S_MAN_WIDTH'({fclass.is_normal, safe_dataa[F64_MAN-1:0]})
                   : S_MAN_WIDTH'({fclass.is_normal, safe_dataa[F32_MAN-1:0]});

    wire signed [S_EXP_WIDTH-1:0] src_bias_s = src_is_d ? S_EXP_WIDTH'(F64_BIAS) : S_EXP_WIDTH'(F32_BIAS);
    wire [SUPER_EXP-1:0] src_exp_raw;
    if (HAS_D) begin : g_srcexp_d
        assign src_exp_raw = src_is_d ? safe_dataa[F64_MAN +: F64_EXP]
                                      : SUPER_EXP'(safe_dataa[F32_MAN +: F32_EXP]);
    end else begin : g_srcexp_s
        assign src_exp_raw = safe_dataa[F32_MAN +: F32_EXP];
    end

    wire i_sign = safe_dataa[is_int64 ? (`VX_CFG_XLEN-1) : 31] && is_signed;
    wire [`VX_CFG_XLEN-1:0] i_mag_raw = i_sign ? (-dataa) : dataa;
    wire [`VX_CFG_XLEN-1:0] i_mag = is_int64 ? i_mag_raw : `VX_CFG_XLEN'(i_mag_raw[31:0]);

    logic [S_MAN_WIDTH-1:0]        unpacked_mant_s0;
    logic signed [S_EXP_WIDTH-1:0] unpacked_exp_s0;
    wire input_sign_s0 = is_itof ? i_sign : input_fp_sgn;

    always_comb begin
        if (is_itof) begin
            unpacked_mant_s0 = S_MAN_WIDTH'(i_mag);
            unpacked_exp_s0  = S_EXP_WIDTH'(`VX_CFG_XLEN - 1);
        end else begin // float source: F2I or F2F
            unpacked_mant_s0 = fp_unpacked_mant;
            unpacked_exp_s0  = S_EXP_WIDTH'(src_exp_raw) - src_bias_s
                             + S_EXP_WIDTH'({1'b0, fclass.is_subnormal});
        end
    end

    // Stage 0 -> Stage 1
    wire is_itof_s1, is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, input_sign_s1, src_d_s1, dst_d_s1;
    wire [INST_FRM_BITS-1:0] frm_s1;
    fclass_t fclass_s1;
    wire [S_MAN_WIDTH-1:0]        unpacked_mant_s1;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s1;

    VX_pipe_register #(
        .DATAW (8 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH),
        .DEPTH (LATENCY > 5)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask),
        .data_in  ({is_itof,    is_ftoi,    f2f,       is_signed,    is_int64,    src_is_d, dst_is_d, input_sign_s0, frm,    fclass,    unpacked_mant_s0, unpacked_exp_s0}),
        .data_out ({is_itof_s1, is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, src_d_s1, dst_d_s1, input_sign_s1, frm_s1, fclass_s1, unpacked_mant_s1, unpacked_exp_s1})
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
    wire is_itof_s2, is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, src_d_s2, dst_d_s2;
    wire [INST_FRM_BITS-1:0] frm_s2;
    fclass_t fclass_s2;
    wire [S_MAN_WIDTH-1:0]        unpacked_mant_s2;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s2;
    wire [LZC_RESULT_WIDTH-1:0]   renorm_shamt_s2;

    VX_pipe_register #(
        .DATAW (9 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH + LZC_RESULT_WIDTH),
        .DEPTH (LATENCY > 4)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && stg2_mask),
        .data_in  ({is_itof_s1, is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, input_sign_s1, mant_is_zero_s1, src_d_s1, dst_d_s1, frm_s1, fclass_s1, unpacked_mant_s1, unpacked_exp_s1, renorm_shamt_s1}),
        .data_out ({is_itof_s2, is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, src_d_s2, dst_d_s2, frm_s2, fclass_s2, unpacked_mant_s2, unpacked_exp_s2, renorm_shamt_s2})
    );

    // ======================================================================
    // Stage 2: Normalize (Left Shift)
    // ======================================================================

    // The significand {hidden, mantissa} occupies the low (1+src_man) bits of
    // the S_MAN_WIDTH field. LZC counts the high padding too: subtract it back
    // for float inputs, using the source format's mantissa width.
    wire [S_EXP_WIDTH-1:0] fp_mant_offset_s2 = S_EXP_WIDTH'(S_MAN_WIDTH - 1)
        - (src_d_s2 ? S_EXP_WIDTH'(F64_MAN) : S_EXP_WIDTH'(F32_MAN));

    wire [S_MAN_WIDTH-1:0]        norm_mant_s2 = unpacked_mant_s2 << renorm_shamt_s2;
    wire signed [S_EXP_WIDTH-1:0] norm_exp_s2  = unpacked_exp_s2 - S_EXP_WIDTH'({1'b0, renorm_shamt_s2})
        + (is_itof_s2 ? S_EXP_WIDTH'(0) : fp_mant_offset_s2);

    // Stage 2 -> Stage 3
    wire is_itof_s3, is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, src_d_s3, dst_d_s3;
    wire [INST_FRM_BITS-1:0] frm_s3;
    fclass_t fclass_s3;
    wire [S_MAN_WIDTH-1:0]        norm_mant_s3;
    wire signed [S_EXP_WIDTH-1:0] norm_exp_s3;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s3; // pre-renorm exp, needed for f2i_shamt

    VX_pipe_register #(
        .DATAW (9 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + 2*S_EXP_WIDTH),
        .DEPTH (LATENCY > 3)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-5]),
        .data_in  ({is_itof_s2, is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, src_d_s2, dst_d_s2, frm_s2, fclass_s2, norm_mant_s2, norm_exp_s2,  unpacked_exp_s2}),
        .data_out ({is_itof_s3, is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, src_d_s3, dst_d_s3, frm_s3, fclass_s3, norm_mant_s3, norm_exp_s3,  unpacked_exp_s3})
    );

    // ======================================================================
    // Stage 3: Calculate Alignment Shift
    // ======================================================================

    // F2I: use pre-renorm exponent to avoid double-shifting
    wire signed [S_EXP_WIDTH-1:0] f2i_shamt_s3 = S_EXP_WIDTH'(S_MAN_WIDTH - 1) - unpacked_exp_s3;

    // Destination float bias and true biased destination exponent.
    wire signed [S_EXP_WIDTH-1:0] dst_bias_s3 = dst_d_s3 ? S_EXP_WIDTH'(F64_BIAS) : S_EXP_WIDTH'(F32_BIAS);
    wire signed [S_EXP_WIDTH-1:0] dst_exp_s3  = norm_exp_s3 + dst_bias_s3;

    // F2F narrowing (FCVT.S.D) can under/overflow the F32 range; widening
    // (FCVT.D.S) is always exact, so it never does. Only the F32-dst finite
    // nonzero case matters here.
    wire f2f_narrow_s3 = is_f2f_s3 & ~dst_d_s3 & ~fclass_s3.is_nan & ~fclass_s3.is_inf & ~fclass_s3.is_zero;
    wire f2f_uf_s3     = f2f_narrow_s3 & (dst_exp_s3 <= 0);                  // -> subnormal/zero
    wire f2f_of_s3     = f2f_narrow_s3 & (dst_exp_s3 >= S_EXP_WIDTH'(255));  // -> overflow to inf
    wire signed [S_EXP_WIDTH-1:0] denorm_sh_s3 = S_EXP_WIDTH'(1) - dst_exp_s3; // right-shift to denormalize

    // Alignment right-shift: F2I integer align, or F2F destination denormalization.
    logic [S_EXP_WIDTH-1:0] align_shamt_s3;
    always_comb begin
        if (is_ftoi_s3) begin // F2I
            align_shamt_s3 = (f2i_shamt_s3 > S_EXP_WIDTH'(S_MAN_WIDTH + 1)) ? S_EXP_WIDTH'(S_MAN_WIDTH + 1) :
                             (f2i_shamt_s3 < 0) ? '0 : f2i_shamt_s3;
        end else if (f2f_uf_s3) begin // F2F underflow -> denormalize into subnormal position
            align_shamt_s3 = (denorm_sh_s3 > S_EXP_WIDTH'(S_MAN_WIDTH + 1)) ? S_EXP_WIDTH'(S_MAN_WIDTH + 1) : denorm_sh_s3;
        end else begin // I2F, normal F2F
            align_shamt_s3 = '0;
        end
    end

    wire [SUPER_EXP-1:0] final_exp_s3 = SUPER_EXP'(dst_exp_s3);

    // Stage 3 -> Stage 4
    wire is_itof_s4, is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, mant_is_zero_s4, src_d_s4, dst_d_s4;
    wire f2f_uf_s4, f2f_of_s4;
    wire [INST_FRM_BITS-1:0] frm_s4;
    fclass_t fclass_s4;
    wire [S_MAN_WIDTH-1:0]  norm_mant_s4;
    wire [S_EXP_WIDTH-1:0]  align_shamt_s4;
    wire [SUPER_EXP-1:0]    final_exp_s4;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s4; // true magnitude exponent (F2I overflow)

    VX_pipe_register #(
        .DATAW (11 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + S_EXP_WIDTH + SUPER_EXP + S_EXP_WIDTH),
        .DEPTH (LATENCY > 2)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-4]),
        .data_in  ({is_itof_s3, is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, src_d_s3, dst_d_s3, f2f_uf_s3, f2f_of_s3, frm_s3, fclass_s3, norm_mant_s3, align_shamt_s3, final_exp_s3, unpacked_exp_s3}),
        .data_out ({is_itof_s4, is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, mant_is_zero_s4, src_d_s4, dst_d_s4, f2f_uf_s4, f2f_of_s4, frm_s4, fclass_s4, norm_mant_s4, align_shamt_s4, final_exp_s4, unpacked_exp_s4})
    );

    // ======================================================================
    // Stage 4: Align & Extract Sticky (Right Shift)
    // ======================================================================

    wire [2*S_MAN_WIDTH:0] aligned_mant_full_s4 = {norm_mant_s4, {(S_MAN_WIDTH+1){1'b0}}} >> align_shamt_s4;

    wire [S_MAN_WIDTH-1:0] aligned_mant_s4 = aligned_mant_full_s4[2*S_MAN_WIDTH : S_MAN_WIDTH+1];
    wire                   guard_bit_s4    = aligned_mant_full_s4[S_MAN_WIDTH];
    wire                   round_bit_s4    = aligned_mant_full_s4[S_MAN_WIDTH-1];
    wire                   sticky_bit_s4   = |aligned_mant_full_s4[S_MAN_WIDTH-2:0];

    // Destination float mantissa width (I2F / F2F)
    wire [5:0] dst_man_w_s4 = dst_d_s4 ? 6'(F64_MAN) : 6'(F32_MAN);

    // Kept significand = {hidden, dst_man}. The round (guard) bit is the first bit
    // BELOW the kept LSB; sticky ORs everything beneath the guard. (Folding the
    // guard into sticky, as a naive low-mask would, breaks RNE on exact ties.)
    wire [6:0] fp_trunc_sh_s4 = 7'(S_MAN_WIDTH - 1) - 7'(dst_man_w_s4);
    wire [S_MAN_WIDTH-1:0] fp_trunc_s4      = aligned_mant_s4 >> fp_trunc_sh_s4;
    wire                   fp_guard_s4      = aligned_mant_s4[LZC_RESULT_WIDTH'(fp_trunc_sh_s4 - 7'd1)];
    wire [S_MAN_WIDTH-1:0] fp_sticky_mask_s4= (S_MAN_WIDTH'(1) << (fp_trunc_sh_s4 - 7'd1)) - 1;
    wire                   sticky_red_s4    = |(aligned_mant_s4 & fp_sticky_mask_s4) | guard_bit_s4 | round_bit_s4 | sticky_bit_s4;

    logic [1:0]            round_sticky_bits_s4;
    logic [S_MAN_WIDTH-1:0] pre_round_abs_s4;

    always_comb begin
        if (is_ftoi_s4) begin // F2I
            // Integer LSB is aligned_mant_full_s4[S_MAN_WIDTH+1], so the first
            // fractional bit below it is guard_bit_s4; sticky ORs everything beneath.
            round_sticky_bits_s4 = {guard_bit_s4, round_bit_s4 | sticky_bit_s4};
            pre_round_abs_s4     = aligned_mant_s4;
        end else begin // I2F, F2F
            round_sticky_bits_s4 = {fp_guard_s4, sticky_red_s4};
            pre_round_abs_s4     = fp_trunc_s4;
        end
    end

    // Stage 4 -> Stage 5
    wire is_itof_s5, is_ftoi_s5, is_f2f_s5, is_signed_s5, is_int64_s5, input_sign_s5, mant_is_zero_s5, src_d_s5, dst_d_s5;
    wire f2f_uf_s5, f2f_of_s5;
    wire [INST_FRM_BITS-1:0] frm_s5;
    fclass_t fclass_s5;
    wire [1:0]            round_sticky_bits_s5;
    wire [S_MAN_WIDTH-1:0] pre_round_abs_s5;
    wire [SUPER_EXP-1:0]  final_exp_s5;
    wire signed [S_EXP_WIDTH-1:0] unpacked_exp_s5;

    VX_pipe_register #(
        .DATAW (11 + INST_FRM_BITS + $bits(fclass_t) + 2 + S_MAN_WIDTH + SUPER_EXP + S_EXP_WIDTH),
        .DEPTH (LATENCY > 1)
    ) pipe_reg5 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-3]),
        .data_in  ({is_itof_s4, is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, mant_is_zero_s4, src_d_s4, dst_d_s4, f2f_uf_s4, f2f_of_s4, frm_s4, fclass_s4, round_sticky_bits_s4, pre_round_abs_s4, final_exp_s4, unpacked_exp_s4}),
        .data_out ({is_itof_s5, is_ftoi_s5, is_f2f_s5, is_signed_s5, is_int64_s5, input_sign_s5, mant_is_zero_s5, src_d_s5, dst_d_s5, f2f_uf_s5, f2f_of_s5, frm_s5, fclass_s5, round_sticky_bits_s5, pre_round_abs_s5, final_exp_s5, unpacked_exp_s5})
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
    // F2F underflow packs a subnormal (base exponent 0; a rounding carry into the
    // hidden bit then bumps it to the smallest normal via f32_man_ovf).
    wire [15:0] safe_dst_exp     = (mant_is_zero_s5 | f2f_uf_s5) ? 16'h0 : final_exp_16;

    // Integer result (F2I)
    wire [63:0] abs_xlen_64  = 64'(rounded_abs_s5[`VX_CFG_XLEN-1:0]);
    wire [63:0] safe_int_res = rounded_sign_s5 ? (-abs_xlen_64) : abs_xlen_64;
    wire [31:0] int_32_res   = safe_int_res[31:0];
    wire [63:0] int_64_res   = safe_int_res;

    // ---- Float pack (I2F / F2F), per destination format ----
    // F32 result, NaN-boxed when XLEN>32. F64 result fills the full width.
    wire f32_man_ovf = rounded_abs_s5[F32_MAN + 1];
    wire [F32_EXP-1:0] fp32_exp = safe_dst_exp[F32_EXP-1:0] + F32_EXP'(f32_man_ovf);
    wire [F32_MAN-1:0] fp32_man = f32_man_ovf ? '0 : safe_rounded_abs[F32_MAN-1:0];
    wire [31:0] fp_32_res = {rounded_sign_s5, fp32_exp, fp32_man};

    wire [`VX_CFG_XLEN-1:0] f32_boxed = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, fp_32_res}
                                                            : `VX_CFG_XLEN'(fp_32_res);

    wire [`VX_CFG_XLEN-1:0] fp_dst_res;
    wire f2f_nv_s5, f2f_nx_s5;
    if (HAS_D) begin : g_fp_pack_d
        wire f64_man_ovf = rounded_abs_s5[F64_MAN + 1];
        wire [F64_EXP-1:0] fp64_exp = safe_dst_exp[F64_EXP-1:0] + F64_EXP'(f64_man_ovf);
        wire [F64_MAN-1:0] fp64_man = f64_man_ovf ? '0 : safe_rounded_abs[F64_MAN-1:0];
        wire [63:0] fp_64_res = {rounded_sign_s5, fp64_exp, fp64_man};

        // Canonical specials for F2F (NaN/Inf/Zero propagation in float domain)
        wire [31:0] f2f_qnan_32 = 32'h7FC00000;
        wire [31:0] f2f_inf_32  = {rounded_sign_s5, 8'hFF, 23'd0};

        wire [63:0] f2f_qnan_64 = 64'h7FF8000000000000;
        wire [63:0] f2f_inf_64  = {rounded_sign_s5, 11'h7FF, 52'd0};
        wire [63:0] f2f_zero_64 = {input_sign_s5, 63'd0};

        // Normal (non-special) destination value by format.
        wire [`VX_CFG_XLEN-1:0] normal_dst = dst_d_s5 ? `VX_CFG_XLEN'(fp_64_res) : f32_boxed;

        // F2F special-case results.
        wire is_nan_src  = is_f2f_s5 & fclass_s5.is_nan;
        wire is_inf_src  = is_f2f_s5 & fclass_s5.is_inf;
        wire is_zero_src = is_f2f_s5 & fclass_s5.is_zero;

        reg [`VX_CFG_XLEN-1:0] dst_r;
        always_comb begin
            if (is_nan_src)
                dst_r = dst_d_s5 ? `VX_CFG_XLEN'(f2f_qnan_64)
                                 : ((`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, f2f_qnan_32} : `VX_CFG_XLEN'(f2f_qnan_32));
            else if (is_inf_src)
                dst_r = dst_d_s5 ? `VX_CFG_XLEN'(f2f_inf_64)
                                 : ((`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, f2f_inf_32} : `VX_CFG_XLEN'(f2f_inf_32));
            else if (is_zero_src)
                dst_r = dst_d_s5 ? `VX_CFG_XLEN'(f2f_zero_64)
                                 : ((`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, input_sign_s5, 31'd0} : `VX_CFG_XLEN'({input_sign_s5, 31'd0}));
            else if (f2f_of_s5) // FCVT.S.D overflow -> F32 inf
                dst_r = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, f2f_inf_32} : `VX_CFG_XLEN'(f2f_inf_32);
            else
                dst_r = normal_dst;
        end
        assign fp_dst_res = dst_r;
        assign f2f_nv_s5  = is_f2f_s5 & fclass_s5.is_signaling;
        assign f2f_nx_s5  = is_f2f_s5 & ~fclass_s5.is_nan & ~fclass_s5.is_inf & ~fclass_s5.is_zero & (|round_sticky_bits_s5);
    end else begin : g_fp_pack_s
        assign fp_dst_res = f32_boxed;
        assign f2f_nv_s5  = 1'b0;
        assign f2f_nx_s5  = 1'b0;
    end

    // F2I saturation values for NaN/Inf inputs
    wire use_neg_sat_s5 = fclass_s5.is_inf && input_sign_s5;
    wire [31:0] nan_inf_32 = use_neg_sat_s5 ? (is_signed_s5 ? 32'h80000000 : 32'h00000000)
                                             : (is_signed_s5 ? 32'h7FFFFFFF : 32'hFFFFFFFF);
    wire [63:0] nan_inf_64 = use_neg_sat_s5 ? (is_signed_s5 ? 64'h8000000000000000 : 64'h0000000000000000)
                                             : (is_signed_s5 ? 64'h7FFFFFFFFFFFFFFF : 64'hFFFFFFFFFFFFFFFF);

    // F2I overflow detection (I32 target)
    wire f2i_s32_pos_ovf = is_signed_s5 && !is_int64_s5 && !input_sign_s5 && (|rounded_abs_s5[S_MAN_WIDTH-1:31]);
    wire f2i_s32_neg_ovf = is_signed_s5 && !is_int64_s5 &&  input_sign_s5 && (rounded_abs_s5 > S_MAN_WIDTH'(32'h80000000));
    wire f2i_u32_neg_ovf = !is_signed_s5 && !is_int64_s5 && rounded_sign_s5 && (|rounded_abs_s5);

    // Unsigned positive overflow: magnitude >= 2^TW (TW=32/64). When S_MAN_WIDTH
    // equals the target width the high bits aren't representable in rounded_abs,
    // so detect from the true exponent; also catch a round-up carry out of the
    // all-ones magnitude (e.g. 2^32-1 rounding up to 2^32).
    wire round_carry_out_s5 = (rounded_abs_s5 == '0) && (|pre_round_abs_s5);
    wire f2i_u32_pos_ovf = !is_signed_s5 && !is_int64_s5 && !input_sign_s5
                         && ((unpacked_exp_s5 >= S_EXP_WIDTH'(32)) || round_carry_out_s5);
    wire f2i_u64_pos_ovf;
    if (`VX_CFG_XLEN == 64) begin : g_f2i_u64pos
        assign f2i_u64_pos_ovf = !is_signed_s5 && is_int64_s5 && !input_sign_s5
                               && ((unpacked_exp_s5 >= S_EXP_WIDTH'(64)) || round_carry_out_s5);
    end else begin : g_f2i_no_u64pos
        assign f2i_u64_pos_ovf = 1'b0;
    end

    // F2I overflow detection (I64 target, XLEN=64 only)
    wire f2i_s64_pos_ovf, f2i_s64_neg_ovf, f2i_u64_neg_ovf;
    if (`VX_CFG_XLEN == 64) begin : g_f2i_64ovf
        assign f2i_s64_pos_ovf = is_signed_s5 && is_int64_s5 && !input_sign_s5 && (|rounded_abs_s5[S_MAN_WIDTH-1:63]);
        assign f2i_s64_neg_ovf = is_signed_s5 && is_int64_s5 &&  input_sign_s5 && (rounded_abs_s5 > S_MAN_WIDTH'(64'h8000000000000000));
        assign f2i_u64_neg_ovf = !is_signed_s5 && is_int64_s5 && rounded_sign_s5 && (|rounded_abs_s5);
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
            // F2I: Float → I32/I64
            if (fclass_s5.is_nan || fclass_s5.is_inf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = is_int64_s5 ? nan_inf_64 : {{32{nan_inf_32[31]}}, nan_inf_32};
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
            end else if (f2i_u32_pos_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = {32'h00000000, 32'hFFFFFFFF};
                res_val_32 = 32'hFFFFFFFF;
            end else if (f2i_u64_pos_ovf) begin
                final_fflags_s5.NV = 1'b1;
                res_val_64 = 64'hFFFFFFFFFFFFFFFF;
                res_val_32 = 32'hFFFFFFFF;
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
                res_val_64 = is_int64_s5 ? int_64_res : {{32{int_32_res[31]}}, int_32_res};
                res_val_32 = int_32_res;
                final_fflags_s5.NX = (|round_sticky_bits_s5);
            end
        end else begin
            // I2F or F2F: float destination
            res_val_64 = 64'(fp_dst_res);
            res_val_32 = fp_dst_res[31:0];
            final_fflags_s5.NV = f2f_nv_s5;
            final_fflags_s5.OF = f2f_of_s5;
            final_fflags_s5.UF = f2f_uf_s5 & (|round_sticky_bits_s5);
            final_fflags_s5.NX = is_f2f_s5 ? (f2f_nx_s5 | f2f_of_s5 | (f2f_uf_s5 & (|round_sticky_bits_s5)))
                                           : (|round_sticky_bits_s5);
        end
    end

    `UNUSED_VAR ({is_itof_s5, src_d_s5, dst_d_s5, fclass_s5, safe_rounded_abs, safe_dst_exp, res_val_64, res_val_32})

    wire [`VX_CFG_XLEN-1:0] final_result_s5;
    if (`VX_CFG_XLEN == 64) begin : g_out_x64
        assign final_result_s5 = res_val_64;
    end else begin : g_out_x32
        assign final_result_s5 = res_val_32;
    end

    // EXCEPT_ENABLE=0 ties fflags to 0.
    fflags_t out_fflags_s5;
    assign out_fflags_s5 = EXCEPT_ENABLE ? final_fflags_s5 : '0;

    // Stage 5 -> Output Register
    VX_pipe_register #(
        .DATAW (`VX_CFG_XLEN + `FP_FLAGS_BITS),
        .DEPTH (OUT_REG)
    ) pipe_reg_out (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-2]),
        .data_in  ({final_result_s5, out_fflags_s5}),
        .data_out ({result,          fflags})
    );

endmodule
