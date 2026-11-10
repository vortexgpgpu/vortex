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
// LATENCY: 5 for XLEN=32, 6 for XLEN=64 (registers the unpacked source
// ahead of the S1 integer negate and LZC).
module VX_fcvt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*;
#(
    parameter LATENCY   = 5,
    parameter FLEN      = 32,
    // Width of the float registers: an F32-only unit (FLEN=32) of a FLEN 64
    // core checks a single-precision source's NaN box against it.
    parameter REGW      = FLEN,
    parameter OUT_REG   = 0,
    // 1: full IEEE subnormal support. 0: flush-to-zero (DAZ subnormal float
    //    sources to signed zero) for area. Use 0 for relaxed paths.
    parameter SUBNORM_ENABLE = 1,
    // 1: produce fflags (NV/OF/UF/NX). 0: tie fflags to 0 (area).
    parameter EXCEPT_ENABLE  = 1,
    parameter HALF_ENABLE    = 0
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
    input wire src_h,
    input wire dst_h,

    input wire [`VX_CFG_XLEN-1:0]  dataa,
    output wire [`VX_CFG_XLEN-1:0] result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam F32_EXP  = 8,  F32_MAN = 23, F32_BIAS = 127;
    localparam F64_EXP  = 11, F64_MAN = 52, F64_BIAS = 1023;
    localparam F16_EXP  = 5,  F16_MAN = 10, F16_BIAS = 15;
    localparam HAS_D    = (FLEN >= 64);
    localparam HAS_H    = (HALF_ENABLE != 0);

    // Widest enabled float format drives the canonical internal datapath.
    localparam SUPER_MAN = HAS_D ? F64_MAN : F32_MAN;
    localparam SUPER_EXP = HAS_D ? F64_EXP : F32_EXP;

    // Internal widths driven by XLEN (for I64 support) and the super mantissa.
    localparam S_MAN_WIDTH      = `MAX(1 + SUPER_MAN, `VX_CFG_XLEN);
    localparam S_EXP_WIDTH      = `MAX(`CLOG2(`VX_CFG_XLEN), SUPER_EXP + 2) + 2;
    localparam LZC_RESULT_WIDTH = `CLOG2(S_MAN_WIDTH);
    localparam ALIGN_SH_WIDTH   = `CLOG2(S_MAN_WIDTH + 2);

    `UNUSED_VAR (src_fmt)
    `UNUSED_VAR (dst_fmt)
    `UNUSED_VAR (is_f2f)
    `UNUSED_VAR (src_h)
    `UNUSED_VAR (dst_h)
    wire src_is_h = HAS_H & src_h;
    wire dst_is_h = HAS_H & dst_h;
    wire src_is_d = HAS_D & src_fmt & ~src_is_h;
    wire dst_is_d = HAS_D & dst_fmt & ~dst_is_h;
    wire f2f      = (HAS_D | HAS_H) & is_f2f;

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

    // A narrower format is NaN-boxed to FLEN; operand bits above FLEN are not part of the FP register.
    wire [15:0] h_op = (&dataa[FLEN-1:16]) ? dataa[15:0] : 16'h7E00;

    wire [31:0] s_op;
    if (REGW > 32) begin : g_s_unbox
        assign s_op = (&dataa[REGW-1:32]) ? dataa[31:0] : 32'h7FC00000;
    end else begin : g_s_raw
        assign s_op = dataa[31:0];
    end

    // Float source field extraction (per source format)
    wire               input_fp_sgn = src_is_h ? h_op[15] : src_is_d ? safe_dataa[63] : s_op[31];

    fclass_t fclass32;
    VX_fp_classifier #(
        .EXP_BITS (F32_EXP),
        .MAN_BITS (F32_MAN)
    ) fp_classifier32 (
        .exp_i  (s_op[F32_MAN +: F32_EXP]),
        .man_i  (s_op[F32_MAN-1:0]),
        .clss_o (fclass32)
    );

    fclass_t fclass_sd;
    if (HAS_D) begin : g_classify_d
        fclass_t fclass64;
        VX_fp_classifier #(
            .EXP_BITS (F64_EXP),
            .MAN_BITS (F64_MAN)
        ) fp_classifier64 (
            .exp_i  (safe_dataa[F64_MAN +: F64_EXP]),
            .man_i  (safe_dataa[F64_MAN-1:0]),
            .clss_o (fclass64)
        );
        assign fclass_sd = src_is_d ? fclass64 : fclass32;
    end else begin : g_classify_s
        assign fclass_sd = fclass32;
    end

    fclass_t fclass;
    if (HAS_H) begin : g_classify_h
        fclass_t fclass16;
        VX_fp_classifier #(
            .EXP_BITS (F16_EXP),
            .MAN_BITS (F16_MAN)
        ) fp_classifier16 (
            .exp_i  (h_op[F16_MAN +: F16_EXP]),
            .man_i  (h_op[F16_MAN-1:0]),
            .clss_o (fclass16)
        );
        assign fclass = src_is_h ? fclass16 : fclass_sd;
    end else begin : g_classify_no_h
        `UNUSED_VAR (h_op)
        assign fclass = fclass_sd;
    end

    // Source significand {hidden, mantissa} placed in the low bits, and the
    // true source exponent (biased − bias). Both reduce to F32 when !HAS_D.
    // DAZ: when subnormals are disabled, a subnormal float source is treated as
    // signed zero (mantissa forced 0 -> downstream zero path).
    wire src_daz = (SUBNORM_ENABLE == 0) & fclass.is_subnormal;
    wire [S_MAN_WIDTH-1:0] fp_unpacked_mant = src_daz ? '0
        : src_is_h ? S_MAN_WIDTH'({fclass.is_normal, h_op[F16_MAN-1:0]})
        : src_is_d ? S_MAN_WIDTH'({fclass.is_normal, safe_dataa[F64_MAN-1:0]})
                   : S_MAN_WIDTH'({fclass.is_normal, s_op[F32_MAN-1:0]});

    wire [SUPER_EXP-1:0] src_exp_sd;
    if (HAS_D) begin : g_srcexp_d
        assign src_exp_sd = src_is_d ? safe_dataa[F64_MAN +: F64_EXP]
                                     : SUPER_EXP'(s_op[F32_MAN +: F32_EXP]);
    end else begin : g_srcexp_s
        assign src_exp_sd = s_op[F32_MAN +: F32_EXP];
    end
    wire [SUPER_EXP-1:0] src_exp_raw = src_is_h ? SUPER_EXP'(h_op[F16_MAN +: F16_EXP]) : src_exp_sd;

    wire i_sign = safe_dataa[is_int64 ? (`VX_CFG_XLEN-1) : 31] && is_signed;
    // A negative integer source is carried as its one's complement y = ~x; the
    // magnitude |x| = y + 1 is formed in stage 1, beside the LZC that reads y.
    wire [`VX_CFG_XLEN-1:0] i_inv_raw = dataa ^ {`VX_CFG_XLEN{i_sign}};
    wire [`VX_CFG_XLEN-1:0] i_inv     = is_int64 ? i_inv_raw : `VX_CFG_XLEN'(i_inv_raw[31:0]);

    // The source exponent is carried biased; stage 1 folds the bias into its
    // per-format constants.
    logic [S_MAN_WIDTH-1:0] mant_raw_s0;
    logic [SUPER_EXP-1:0]   exp_raw_s0;
    logic                   exp_sub_s0;
    wire input_sign_s0 = is_itof ? i_sign : input_fp_sgn;
    wire mant_neg_s0   = is_itof && i_sign;

    always_comb begin
        if (is_itof) begin
            mant_raw_s0 = S_MAN_WIDTH'(i_inv);
            exp_raw_s0  = '0;
            exp_sub_s0  = 1'b0;
        end else begin // float source: F2I or F2F
            mant_raw_s0 = fp_unpacked_mant;
            exp_raw_s0  = src_exp_raw;
            exp_sub_s0  = fclass.is_subnormal;
        end
    end

    // Stage 0 -> Stage 1
    wire is_itof_s1, is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, input_sign_s1, mant_neg_s1, src_d_s1, dst_d_s1, src_h_s1, dst_h_s1;
    wire [INST_FRM_BITS-1:0] frm_s1;
    fclass_t fclass_s1;
    wire [S_MAN_WIDTH-1:0]        mant_raw_s1;
    wire [SUPER_EXP-1:0]          exp_raw_s1;
    wire                          exp_sub_s1;

    VX_pipe_register #(
        .DATAW (12 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + SUPER_EXP),
        .DEPTH (LATENCY > 5)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask),
        .data_in  ({is_itof,    is_ftoi,    f2f,       is_signed,    is_int64,    src_is_d, dst_is_d, src_is_h, dst_is_h, input_sign_s0, mant_neg_s0, frm,    fclass,    mant_raw_s0, exp_raw_s0, exp_sub_s0}),
        .data_out ({is_itof_s1, is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, src_d_s1, dst_d_s1, src_h_s1, dst_h_s1, input_sign_s1, mant_neg_s1, frm_s1, fclass_s1, mant_raw_s1, exp_raw_s1, exp_sub_s1})
    );

    // ======================================================================
    // Stage 1: Integer magnitude, Leading Zero Count (LZC), exponent offsets
    // ======================================================================

    wire [S_MAN_WIDTH-1:0] unpacked_mant_s1 = mant_raw_s1 + S_MAN_WIDTH'(mant_neg_s1);

    // |x| = y + 1 leads where y does, except when y is all ones below its
    // leading one (zero included), where |x| leads one place higher.
    wire [LZC_RESULT_WIDTH-1:0] lzc_raw_s1;
    wire                        lzc_valid_s1;
    VX_lzc #(
        .N (S_MAN_WIDTH)
    ) lzc (
        .data_in   (mant_raw_s1),
        .data_out  (lzc_raw_s1),
        .valid_out (lzc_valid_s1)
    );
    wire lzc_lowmask_s1 = ~(| ((mant_raw_s1 >> 1) & ~mant_raw_s1));

    wire [LZC_RESULT_WIDTH-1:0] renorm_shamt_s1 = ~mant_neg_s1  ? lzc_raw_s1
                                                : ~lzc_valid_s1 ? LZC_RESULT_WIDTH'(S_MAN_WIDTH - 1)
                                                                : (lzc_raw_s1 - LZC_RESULT_WIDTH'(lzc_lowmask_s1));
    wire mant_is_zero_s1 = ~(lzc_valid_s1 || mant_neg_s1);

`ifdef SIMULATION
    wire [LZC_RESULT_WIDTH-1:0] ref_shamt_s1;
    wire                        ref_nonzero_s1;
    VX_lzc #(
        .N (S_MAN_WIDTH)
    ) lzc_ref (
        .data_in   (unpacked_mant_s1),
        .data_out  (ref_shamt_s1),
        .valid_out (ref_nonzero_s1)
    );
    `RUNTIME_ASSERT (~(enable && stg2_mask)
                  || ((mant_is_zero_s1 != ref_nonzero_s1) && (~ref_nonzero_s1 || (renorm_shamt_s1 == ref_shamt_s1))),
        ("%t: fcvt leading-zero count differs from the magnitude's", $time))
`endif

    // The true source exponent is ue = exp_raw + exp_sub + C, with C = -bias
    // for a float source and XLEN-1 for an integer (exp_raw = exp_sub = 0).
    // The destination exponent is ue - renorm_shamt + K, with K the source's
    // mantissa offset (float sources) plus the destination bias. C and K are
    // constants of the (source, destination) formats, folded into one table
    // per use, so that each use is a single add to exp_raw + exp_sub here and
    // stage 2 only subtracts the shift amount.
    localparam integer SRC_OFF_S = S_MAN_WIDTH - 1 - F32_MAN;
    localparam integer SRC_OFF_D = S_MAN_WIDTH - 1 - F64_MAN;
    localparam integer SRC_OFF_H = S_MAN_WIDTH - 1 - F16_MAN;
    wire [1:0] src_sel_s1 = is_itof_s1 ? 2'd3 : (src_h_s1 ? 2'd2 : (src_d_s1 ? 2'd1 : 2'd0));
    wire [1:0] dst_sel_s1 = dst_h_s1 ? 2'd2 : (dst_d_s1 ? 2'd1 : 2'd0);
    wire [3:0][2:0][S_EXP_WIDTH-1:0] exp_k_tbl, exp_dk_tbl;
    wire [3:0][S_EXP_WIDTH-1:0] f2i_sh_tbl, f2i_lo_tbl, ge32_tbl, ge64_tbl;
    for (genvar s = 0; s < 4; ++s) begin : g_exp_src
        localparam integer SRC_OFF = (s == 3) ? 0 : ((s == 2) ? SRC_OFF_H : ((s == 1) ? SRC_OFF_D : SRC_OFF_S));
        localparam integer C_UE    = (s == 3) ? (`VX_CFG_XLEN - 1) : -((s == 2) ? F16_BIAS : ((s == 1) ? F64_BIAS : F32_BIAS));
        for (genvar d = 0; d < 3; ++d) begin : g_exp_dst
            localparam integer DST_BIAS = (d == 2) ? F16_BIAS : ((d == 1) ? F64_BIAS : F32_BIAS);
            assign exp_k_tbl[s][d]  = S_EXP_WIDTH'(C_UE + SRC_OFF + DST_BIAS);
            assign exp_dk_tbl[s][d] = S_EXP_WIDTH'(1 - C_UE - SRC_OFF - DST_BIAS);
        end
        assign f2i_sh_tbl[s] = S_EXP_WIDTH'(S_MAN_WIDTH - 1 - C_UE);  // F2I shift = this - (ue - C)
        assign f2i_lo_tbl[s] = S_EXP_WIDTH'(-2 - C_UE);               // F2I shift > S_MAN_WIDTH + 1 below this
        assign ge32_tbl[s]   = S_EXP_WIDTH'(32 - C_UE);
        assign ge64_tbl[s]   = S_EXP_WIDTH'(64 - C_UE);
    end
    wire signed [S_EXP_WIDTH-1:0] exp_s1 = S_EXP_WIDTH'(exp_raw_s1) + S_EXP_WIDTH'(exp_sub_s1); // ue - C

    wire signed [S_EXP_WIDTH-1:0] exp_k_s1  = exp_k_tbl[src_sel_s1][dst_sel_s1];
    wire signed [S_EXP_WIDTH-1:0] exp_dk_s1 = exp_dk_tbl[src_sel_s1][dst_sel_s1];
    wire signed [S_EXP_WIDTH-1:0] dst_exp_base_s1   = exp_s1 + exp_k_s1;   // dst_exp + renorm_shamt
    wire signed [S_EXP_WIDTH-1:0] denorm_sh_base_s1 = exp_dk_s1 - exp_s1;  // 1 - dst_exp - renorm_shamt

    // F2I: alignment (S_MAN_WIDTH - 1 - ue, clamped to [0, S_MAN_WIDTH + 1])
    // from the pre-renorm exponent, to avoid double-shifting.
    wire signed [S_EXP_WIDTH-1:0] f2i_sh_k_s1  = f2i_sh_tbl[src_sel_s1];
    wire signed [S_EXP_WIDTH-1:0] f2i_lo_k_s1  = f2i_lo_tbl[src_sel_s1];
    wire [ALIGN_SH_WIDTH-1:0]     f2i_shamt_s1 = ALIGN_SH_WIDTH'(f2i_sh_k_s1 - exp_s1);
    wire [ALIGN_SH_WIDTH-1:0] f2i_align_s1 = (exp_s1 < f2i_lo_k_s1) ? ALIGN_SH_WIDTH'(S_MAN_WIDTH + 1) :
                                             (exp_s1 > f2i_sh_k_s1) ? '0 : f2i_shamt_s1;
    // F2I overflow from the true magnitude exponent: ue >= 32, ue >= 64.
    wire exp_ge32_s1 = (exp_s1 >= $signed(ge32_tbl[src_sel_s1]));
    wire exp_ge64_s1 = (exp_s1 >= $signed(ge64_tbl[src_sel_s1]));

    // F2F narrowing (FCVT.S.D, FCVT.H.S, FCVT.H.D) can under/overflow the
    // destination range; widening is always exact, so it never does. Only the
    // F32/F16-dst finite nonzero case matters here.
    wire f2f_narrow_s1 = is_f2f_s1 & ~dst_d_s1 & ~fclass_s1.is_nan & ~fclass_s1.is_inf & ~fclass_s1.is_zero;

    // Stage 1 -> Stage 2
    wire is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, dst_d_s2, dst_h_s2;
    wire f2f_narrow_s2, exp_ge32_s2, exp_ge64_s2;
    wire [INST_FRM_BITS-1:0] frm_s2;
    fclass_t fclass_s2;
    wire [S_MAN_WIDTH-1:0]        unpacked_mant_s2;
    wire [LZC_RESULT_WIDTH-1:0]   renorm_shamt_s2;
    wire signed [S_EXP_WIDTH-1:0] dst_exp_base_s2, denorm_sh_base_s2;
    wire [ALIGN_SH_WIDTH-1:0]     f2i_align_s2;

    VX_pipe_register #(
        .DATAW (11 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + LZC_RESULT_WIDTH + 2*S_EXP_WIDTH + ALIGN_SH_WIDTH),
        .DEPTH (LATENCY > 4)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && stg2_mask),
        .data_in  ({is_ftoi_s1, is_f2f_s1, is_signed_s1, is_int64_s1, input_sign_s1, mant_is_zero_s1, dst_d_s1, dst_h_s1, f2f_narrow_s1, exp_ge32_s1, exp_ge64_s1, frm_s1, fclass_s1, unpacked_mant_s1, renorm_shamt_s1, dst_exp_base_s1, denorm_sh_base_s1, f2i_align_s1}),
        .data_out ({is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, dst_d_s2, dst_h_s2, f2f_narrow_s2, exp_ge32_s2, exp_ge64_s2, frm_s2, fclass_s2, unpacked_mant_s2, renorm_shamt_s2, dst_exp_base_s2, denorm_sh_base_s2, f2i_align_s2})
    );

    // ======================================================================
    // Stage 2: Normalize (Left Shift) & Alignment Shift Amount
    // ======================================================================

    wire [S_MAN_WIDTH-1:0]        norm_mant_s2 = unpacked_mant_s2 << renorm_shamt_s2;
    wire signed [S_EXP_WIDTH-1:0] dst_exp_s2   = dst_exp_base_s2 - S_EXP_WIDTH'({1'b0, renorm_shamt_s2});
    wire signed [S_EXP_WIDTH-1:0] denorm_sh_s2 = denorm_sh_base_s2 + S_EXP_WIDTH'({1'b0, renorm_shamt_s2}); // right-shift to denormalize
    wire signed [S_EXP_WIDTH-1:0] dst_inf_exp_s2 = dst_h_s2 ? S_EXP_WIDTH'(31) : S_EXP_WIDTH'(255);

    wire f2f_uf_s2    = f2f_narrow_s2 & (dst_exp_s2 <= 0);              // -> subnormal/zero
    wire f2f_of_s2    = f2f_narrow_s2 & (dst_exp_s2 >= dst_inf_exp_s2); // -> overflow
    wire f2f_tiny0_s2 = f2f_narrow_s2 & (dst_exp_s2 == 0);

    // Alignment right-shift: F2I integer align, or F2F destination denormalization.
    logic [ALIGN_SH_WIDTH-1:0] align_shamt_s2;
    always_comb begin
        if (is_ftoi_s2) begin // F2I
            align_shamt_s2 = f2i_align_s2;
        end else if (f2f_uf_s2) begin // F2F underflow -> denormalize into subnormal position
            align_shamt_s2 = (denorm_sh_s2 > S_EXP_WIDTH'(S_MAN_WIDTH + 1)) ? ALIGN_SH_WIDTH'(S_MAN_WIDTH + 1) : ALIGN_SH_WIDTH'(denorm_sh_s2);
        end else begin // I2F, normal F2F
            align_shamt_s2 = '0;
        end
    end

    wire [SUPER_EXP-1:0] final_exp_s2 = SUPER_EXP'(dst_exp_s2);

    // Stage 2 -> Stage 3
    wire is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, dst_d_s3, dst_h_s3;
    wire f2f_uf_s3, f2f_of_s3, f2f_tiny0_s3, exp_ge32_s3, exp_ge64_s3;
    wire [INST_FRM_BITS-1:0] frm_s3;
    fclass_t fclass_s3;
    wire [S_MAN_WIDTH-1:0]    norm_mant_s3;
    wire [ALIGN_SH_WIDTH-1:0] align_shamt_s3;
    wire [SUPER_EXP-1:0]      final_exp_s3;

    VX_pipe_register #(
        .DATAW (13 + INST_FRM_BITS + $bits(fclass_t) + S_MAN_WIDTH + ALIGN_SH_WIDTH + SUPER_EXP),
        .DEPTH (LATENCY > 3)
    ) pipe_reg3 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-5]),
        .data_in  ({is_ftoi_s2, is_f2f_s2, is_signed_s2, is_int64_s2, input_sign_s2, mant_is_zero_s2, dst_d_s2, dst_h_s2, f2f_uf_s2, f2f_of_s2, f2f_tiny0_s2, exp_ge32_s2, exp_ge64_s2, frm_s2, fclass_s2, norm_mant_s2, align_shamt_s2, final_exp_s2}),
        .data_out ({is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, dst_d_s3, dst_h_s3, f2f_uf_s3, f2f_of_s3, f2f_tiny0_s3, exp_ge32_s3, exp_ge64_s3, frm_s3, fclass_s3, norm_mant_s3, align_shamt_s3, final_exp_s3})
    );

    // ======================================================================
    // Stage 3: Align & Extract Sticky (Right Shift)
    // ======================================================================

    wire [2*S_MAN_WIDTH:0] aligned_mant_full_s3 = {norm_mant_s3, {(S_MAN_WIDTH+1){1'b0}}} >> align_shamt_s3;

    wire [S_MAN_WIDTH-1:0] aligned_mant_s3 = aligned_mant_full_s3[2*S_MAN_WIDTH : S_MAN_WIDTH+1];
    wire                   guard_bit_s3    = aligned_mant_full_s3[S_MAN_WIDTH];
    wire                   round_bit_s3    = aligned_mant_full_s3[S_MAN_WIDTH-1];
    wire                   sticky_bit_s3   = |aligned_mant_full_s3[S_MAN_WIDTH-2:0];

    // Kept significand = {hidden, dst_man}. The round (guard) bit is the first bit
    // BELOW the kept LSB; sticky ORs everything beneath the guard. (Folding the
    // guard into sticky, as a naive low-mask would, breaks RNE on exact ties.)
    // The kept width is a constant of the destination format (0=F32, 1=F64,
    // 2=F16), so each format's split is wiring and the format selects one.
    wire [S_MAN_WIDTH:0] rnd_ext_s3  = {aligned_mant_s3, guard_bit_s3};
    wire [S_MAN_WIDTH:0] fine_ext_s3 = {norm_mant_s3, 1'b0};

    wire [2:0][S_MAN_WIDTH-1:0] fmt_trunc_s3;
    wire [2:0] fmt_guard_s3, fmt_sticky_s3, fmt_fine_s3;
    wire [2:0][1:0] fmt_fine_rs_s3;
    for (genvar f = 0; f < 3; ++f) begin : g_fmt
        localparam integer MAN = (f == 1) ? F64_MAN : ((f == 2) ? F16_MAN : F32_MAN);
        localparam integer TSH = S_MAN_WIDTH - 1 - MAN; // bits below the kept LSB
        if (TSH >= 0) begin : g_valid
            assign fmt_trunc_s3[f] = aligned_mant_s3 >> TSH;
            assign fmt_guard_s3[f] = rnd_ext_s3[TSH];
            assign fmt_fine_s3[f]  = &norm_mant_s3[S_MAN_WIDTH-1:TSH];
            assign fmt_fine_rs_s3[f][1] = fine_ext_s3[TSH];
            if (TSH > 0) begin : g_sticky
                assign fmt_sticky_s3[f]     = |rnd_ext_s3[TSH-1:0];
                assign fmt_fine_rs_s3[f][0] = |fine_ext_s3[TSH-1:0];
            end else begin : g_no_sticky
                assign fmt_sticky_s3[f]     = 1'b0;
                assign fmt_fine_rs_s3[f][0] = 1'b0;
            end
        end else begin : g_none
            assign fmt_trunc_s3[f]   = '0;
            assign fmt_guard_s3[f]   = 1'b0;
            assign fmt_sticky_s3[f]  = 1'b0;
            assign fmt_fine_s3[f]    = 1'b0;
            assign fmt_fine_rs_s3[f] = '0;
        end
    end

    wire [1:0] dst_fmt_s3 = dst_h_s3 ? 2'd2 : (dst_d_s3 ? 2'd1 : 2'd0);

    wire [S_MAN_WIDTH-1:0] fp_trunc_s3   = fmt_trunc_s3[dst_fmt_s3];
    wire                   fp_guard_s3   = fmt_guard_s3[dst_fmt_s3];
    wire                   sticky_red_s3 = fmt_sticky_s3[dst_fmt_s3] | round_bit_s3 | sticky_bit_s3;
    wire [2:0]             fine_round_s3 = {fmt_fine_s3[dst_fmt_s3], fmt_fine_rs_s3[dst_fmt_s3]};

    logic [1:0]            round_sticky_bits_s3;
    logic [S_MAN_WIDTH-1:0] pre_round_abs_s3;

    always_comb begin
        if (is_ftoi_s3) begin // F2I
            // Integer LSB is aligned_mant_full_s3[S_MAN_WIDTH+1], so the first
            // fractional bit below it is guard_bit_s3; sticky ORs everything beneath.
            round_sticky_bits_s3 = {guard_bit_s3, round_bit_s3 | sticky_bit_s3};
            pre_round_abs_s3     = aligned_mant_s3;
        end else begin // I2F, F2F
            round_sticky_bits_s3 = {fp_guard_s3, sticky_red_s3};
            pre_round_abs_s3     = fp_trunc_s3;
        end
    end

    // Stage 3 -> Stage 4
    wire is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, mant_is_zero_s4, dst_d_s4, dst_h_s4;
    wire f2f_uf_s4, f2f_of_s4, f2f_tiny0_s4, exp_ge32_s4, exp_ge64_s4;
    wire [2:0] fine_round_s4;
    wire [INST_FRM_BITS-1:0] frm_s4;
    fclass_t fclass_s4;
    wire [1:0]             round_sticky_bits_s4;
    wire [S_MAN_WIDTH-1:0] pre_round_abs_s4;
    wire [SUPER_EXP-1:0]   final_exp_s4;

    VX_pipe_register #(
        .DATAW (16 + INST_FRM_BITS + $bits(fclass_t) + 2 + S_MAN_WIDTH + SUPER_EXP),
        .DEPTH (LATENCY > 2)
    ) pipe_reg4 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-4]),
        .data_in  ({is_ftoi_s3, is_f2f_s3, is_signed_s3, is_int64_s3, input_sign_s3, mant_is_zero_s3, dst_d_s3, dst_h_s3, f2f_uf_s3, f2f_of_s3, f2f_tiny0_s3, exp_ge32_s3, exp_ge64_s3, fine_round_s3, frm_s3, fclass_s3, round_sticky_bits_s3, pre_round_abs_s3, final_exp_s3}),
        .data_out ({is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, mant_is_zero_s4, dst_d_s4, dst_h_s4, f2f_uf_s4, f2f_of_s4, f2f_tiny0_s4, exp_ge32_s4, exp_ge64_s4, fine_round_s4, frm_s4, fclass_s4, round_sticky_bits_s4, pre_round_abs_s4, final_exp_s4})
    );

    // ======================================================================
    // Stage 4: Rounding
    // ======================================================================

    wire round_up_s4;
    wire rounded_sign_s4;

    VX_fp_rounding #(
        .DAT_WIDTH (S_MAN_WIDTH)
    ) fp_rounding (
        .abs_value_i         (pre_round_abs_s4),
        .sign_i              (input_sign_s4),
        .round_sticky_bits_i (round_sticky_bits_s4),
        .rnd_mode_i          (frm_s4),
        .effective_subtraction_i (1'b0),
        `UNUSED_PIN          (abs_rounded_o),
        .sign_o              (rounded_sign_s4),
        `UNUSED_PIN          (exact_zero_o),
        .round_up_o          (round_up_s4)
    );

    // The increment runs beside the round-up decision, which then selects.
    wire [S_MAN_WIDTH-1:0] pre_inc_s4     = pre_round_abs_s4 + S_MAN_WIDTH'(1);
    wire [S_MAN_WIDTH-1:0] rounded_abs_s4 = round_up_s4 ? pre_inc_s4 : pre_round_abs_s4;

    // F2I result -(pre + round_up): ~pre when rounding up, else -pre.
    wire [`VX_CFG_XLEN-1:0] pre_xlen_s4 = pre_round_abs_s4[`VX_CFG_XLEN-1:0];
    wire [`VX_CFG_XLEN-1:0] pre_neg_s4  = -pre_xlen_s4;
    wire [`VX_CFG_XLEN-1:0] int_res_s4  = rounded_sign_s4 ? (round_up_s4 ? ~pre_xlen_s4 : pre_neg_s4)
                                                          : rounded_abs_s4[`VX_CFG_XLEN-1:0];
    wire [S_MAN_WIDTH-1:0]  res_mag_s4  = is_ftoi_s4 ? S_MAN_WIDTH'(int_res_s4) : rounded_abs_s4;

    // F2I range checks of the rounded magnitude pre + round_up (wrapping to 0
    // from all ones), resolved from the unrounded one so that they run beside
    // the increment, not after it. A magnitude of 2^S_MAN_WIDTH or more aligns
    // unshifted, so the signed negative checks also take the true exponent.
    wire pre_all1_s4 = &pre_round_abs_s4;
    wire rnd_nz_s4   = round_up_s4 ? ~pre_all1_s4 : (|pre_round_abs_s4); // rounded != 0

    // rounded >= 2^K
    function automatic logic rnd_ge_pow2(input logic [S_MAN_WIDTH-1:0] pre, input logic ru, input logic all1, input integer K);
        logic hi, lo1;
        hi  = |(pre >> K);
        lo1 = &(pre | ~((S_MAN_WIDTH'(1) << K) - S_MAN_WIDTH'(1)));
        rnd_ge_pow2 = ru ? ((hi | lo1) & ~all1) : hi;
    endfunction
    // rounded > 2^K
    function automatic logic rnd_gt_pow2(input logic [S_MAN_WIDTH-1:0] pre, input logic ru, input logic all1, input integer K);
        logic hi, at, lo0;
        hi  = |(pre >> (K + 1));
        at  = pre[K];
        lo0 = |(pre & ((S_MAN_WIDTH'(1) << K) - S_MAN_WIDTH'(1)));
        rnd_gt_pow2 = ru ? ((hi | at) & ~all1) : (hi | (at & lo0));
    endfunction

    wire f2i_s32_pos_ovf_s4 = is_signed_s4 && !is_int64_s4 && !input_sign_s4 && rnd_ge_pow2(pre_round_abs_s4, round_up_s4, pre_all1_s4, 31);
    wire f2i_s32_neg_ovf_s4 = is_signed_s4 && !is_int64_s4 &&  input_sign_s4
                           && (exp_ge32_s4 || rnd_gt_pow2(pre_round_abs_s4, round_up_s4, pre_all1_s4, 31));
    wire f2i_u32_neg_ovf_s4 = !is_signed_s4 && !is_int64_s4 && rounded_sign_s4 && rnd_nz_s4;

    // Unsigned positive overflow: magnitude >= 2^TW (TW=32/64). When S_MAN_WIDTH
    // equals the target width the high bits aren't representable in rounded_abs,
    // so detect from the true exponent; also catch a round-up carry out of the
    // all-ones magnitude (e.g. 2^32-1 rounding up to 2^32).
    wire round_carry_out_s4 = round_up_s4 && pre_all1_s4;
    wire f2i_u32_pos_ovf_s4 = !is_signed_s4 && !is_int64_s4 && !input_sign_s4
                           && (exp_ge32_s4 || round_carry_out_s4 || rnd_ge_pow2(pre_round_abs_s4, round_up_s4, pre_all1_s4, 32));
    wire f2i_u64_pos_ovf_s4;
    if (`VX_CFG_XLEN == 64) begin : g_f2i_u64pos
        assign f2i_u64_pos_ovf_s4 = !is_signed_s4 && is_int64_s4 && !input_sign_s4 && (exp_ge64_s4 || round_carry_out_s4);
    end else begin : g_f2i_no_u64pos
        `UNUSED_VAR (exp_ge64_s4)
        assign f2i_u64_pos_ovf_s4 = 1'b0;
    end

    // F2I overflow detection (I64 target, XLEN=64 only)
    wire f2i_s64_pos_ovf_s4, f2i_s64_neg_ovf_s4, f2i_u64_neg_ovf_s4;
    if (`VX_CFG_XLEN == 64) begin : g_f2i_64ovf
        assign f2i_s64_pos_ovf_s4 = is_signed_s4 && is_int64_s4 && !input_sign_s4 && rnd_ge_pow2(pre_round_abs_s4, round_up_s4, pre_all1_s4, 63);
        assign f2i_s64_neg_ovf_s4 = is_signed_s4 && is_int64_s4 &&  input_sign_s4
                                 && (exp_ge64_s4 || rnd_gt_pow2(pre_round_abs_s4, round_up_s4, pre_all1_s4, 63));
        assign f2i_u64_neg_ovf_s4 = !is_signed_s4 && is_int64_s4 && rounded_sign_s4 && rnd_nz_s4;
    end else begin : g_f2i_no64ovf
        assign f2i_s64_pos_ovf_s4 = 1'b0;
        assign f2i_s64_neg_ovf_s4 = 1'b0;
        assign f2i_u64_neg_ovf_s4 = 1'b0;
    end

    wire [7:0] f2i_ovf_s4 = {f2i_s32_pos_ovf_s4, f2i_s32_neg_ovf_s4, f2i_u32_neg_ovf_s4, f2i_u32_pos_ovf_s4,
                             f2i_u64_pos_ovf_s4, f2i_s64_pos_ovf_s4, f2i_s64_neg_ovf_s4, f2i_u64_neg_ovf_s4};
    wire rs_any_s4 = |round_sticky_bits_s4;

    // Stage 4 -> Stage 5
    wire is_ftoi_s5, is_f2f_s5, is_signed_s5, is_int64_s5, input_sign_s5, rounded_sign_s5, mant_is_zero_s5, dst_d_s5, dst_h_s5;
    wire f2f_uf_s5, f2f_of_s5, f2f_tiny0_s5, rs_any_s5;
    wire [2:0] fine_round_s5;
    wire [INST_FRM_BITS-1:0] frm_s5;
    fclass_t fclass_s5;
    wire [7:0]             f2i_ovf_s5;
    wire [S_MAN_WIDTH-1:0] res_mag_s5;
    wire [SUPER_EXP-1:0]   final_exp_s5;

    VX_pipe_register #(
        .DATAW (16 + INST_FRM_BITS + $bits(fclass_t) + 8 + S_MAN_WIDTH + SUPER_EXP),
        .DEPTH (LATENCY > 1)
    ) pipe_reg5 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-3]),
        .data_in  ({is_ftoi_s4, is_f2f_s4, is_signed_s4, is_int64_s4, input_sign_s4, rounded_sign_s4, mant_is_zero_s4, dst_d_s4, dst_h_s4, f2f_uf_s4, f2f_of_s4, f2f_tiny0_s4, rs_any_s4, fine_round_s4, frm_s4, fclass_s4, f2i_ovf_s4, res_mag_s4, final_exp_s4}),
        .data_out ({is_ftoi_s5, is_f2f_s5, is_signed_s5, is_int64_s5, input_sign_s5, rounded_sign_s5, mant_is_zero_s5, dst_d_s5, dst_h_s5, f2f_uf_s5, f2f_of_s5, f2f_tiny0_s5, rs_any_s5, fine_round_s5, frm_s5, fclass_s5, f2i_ovf_s5, res_mag_s5, final_exp_s5})
    );

    // ======================================================================
    // Stage 5: Pack & Select
    // ======================================================================

    wire [S_MAN_WIDTH-1:0] rounded_abs_s5 = res_mag_s5;
    wire f2i_s32_pos_ovf, f2i_s32_neg_ovf, f2i_u32_neg_ovf, f2i_u32_pos_ovf;
    wire f2i_u64_pos_ovf, f2i_s64_pos_ovf, f2i_s64_neg_ovf, f2i_u64_neg_ovf;
    assign {f2i_s32_pos_ovf, f2i_s32_neg_ovf, f2i_u32_neg_ovf, f2i_u32_pos_ovf,
            f2i_u64_pos_ovf, f2i_s64_pos_ovf, f2i_s64_neg_ovf, f2i_u64_neg_ovf} = f2i_ovf_s5;

    wire [63:0] safe_rounded_abs = 64'(rounded_abs_s5);
    wire [15:0] final_exp_16     = 16'(final_exp_s5);
    // F2F underflow packs a subnormal (base exponent 0; a rounding carry into the
    // hidden bit then bumps it to the smallest normal via f32_man_ovf).
    wire [15:0] safe_dst_exp     = (mant_is_zero_s5 | f2f_uf_s5) ? 16'h0 : final_exp_16;

    // Integer result (F2I)
    wire [63:0] safe_int_res = 64'(res_mag_s5[`VX_CFG_XLEN-1:0]); // signed in stage 4
    wire [31:0] int_32_res   = safe_int_res[31:0];
    wire [63:0] int_64_res   = safe_int_res;

    // ---- Float pack (I2F / F2F), per destination format ----
    // F32 result, NaN-boxed when XLEN>32. F64 result fills the full width.
    wire f32_man_ovf = f2f_uf_s5 ? rounded_abs_s5[F32_MAN] : rounded_abs_s5[F32_MAN + 1];
    wire [F32_EXP-1:0] fp32_exp = safe_dst_exp[F32_EXP-1:0] + F32_EXP'(f32_man_ovf);
    wire [F32_MAN-1:0] fp32_man = f32_man_ovf ? '0 : safe_rounded_abs[F32_MAN-1:0];
    wire [31:0] fp_32_res = {rounded_sign_s5, fp32_exp, fp32_man};

    wire [`VX_CFG_XLEN-1:0] f32_boxed = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, fp_32_res}
                                                            : `VX_CFG_XLEN'(fp_32_res);

    wire [`VX_CFG_XLEN-1:0] fp_dst_res;
    wire f2f_nv_s5, f2f_nx_s5, f2f_of_out_s5, f2f_uf_out_s5;
    if (HAS_D || HAS_H) begin : g_fp_pack_f2f
        wire [`VX_CFG_XLEN-1:0] fp64_boxed;
        if (HAS_D) begin : g_fp64
            wire f64_man_ovf = rounded_abs_s5[F64_MAN + 1];
            wire [F64_EXP-1:0] fp64_exp = safe_dst_exp[F64_EXP-1:0] + F64_EXP'(f64_man_ovf);
            wire [F64_MAN-1:0] fp64_man = f64_man_ovf ? '0 : safe_rounded_abs[F64_MAN-1:0];
            assign fp64_boxed = `VX_CFG_XLEN'({rounded_sign_s5, fp64_exp, fp64_man});
        end else begin : g_no_fp64
            assign fp64_boxed = '0;
        end

        wire f16_man_ovf = f2f_uf_s5 ? rounded_abs_s5[F16_MAN] : rounded_abs_s5[F16_MAN + 1];
        wire [F16_EXP-1:0] fp16_exp = safe_dst_exp[F16_EXP-1:0] + F16_EXP'(f16_man_ovf);
        wire [F16_MAN-1:0] fp16_man = f16_man_ovf ? '0 : safe_rounded_abs[F16_MAN-1:0];
        wire [15:0] fp_16_res = {rounded_sign_s5, fp16_exp, fp16_man};

        wire [`VX_CFG_XLEN-1:0] qnan_d = `VX_CFG_XLEN'(64'h7FF8000000000000);
        wire [`VX_CFG_XLEN-1:0] inf_d  = `VX_CFG_XLEN'({input_sign_s5, 11'h7FF, 52'd0});
        wire [`VX_CFG_XLEN-1:0] zero_d = `VX_CFG_XLEN'({input_sign_s5, 63'd0});

        wire [31:0] qnan_s32 = 32'h7FC00000;
        wire [31:0] inf_s32  = {input_sign_s5, 8'hFF, 23'd0};
        wire [31:0] zero_s32 = {input_sign_s5, 31'd0};
        wire [31:0] max_s32  = {input_sign_s5, 8'hFE, {23{1'b1}}};

        wire [15:0] qnan_h16 = 16'h7E00;
        wire [15:0] inf_h16  = {input_sign_s5, 5'h1F, 10'd0};
        wire [15:0] zero_h16 = {input_sign_s5, 15'd0};
        wire [15:0] max_h16  = {input_sign_s5, 5'h1E, {10{1'b1}}};

        wire [`VX_CFG_XLEN-1:0] box_s_qnan = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, qnan_s32} : `VX_CFG_XLEN'(qnan_s32);
        wire [`VX_CFG_XLEN-1:0] box_s_inf  = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, inf_s32}  : `VX_CFG_XLEN'(inf_s32);
        wire [`VX_CFG_XLEN-1:0] box_s_zero = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, zero_s32} : `VX_CFG_XLEN'(zero_s32);
        wire [`VX_CFG_XLEN-1:0] box_s_max  = (`VX_CFG_XLEN > 32) ? {{(`VX_CFG_XLEN-32){1'b1}}, max_s32}  : `VX_CFG_XLEN'(max_s32);

        wire [`VX_CFG_XLEN-1:0] box_h_res  = {{(`VX_CFG_XLEN-16){1'b1}}, fp_16_res};
        wire [`VX_CFG_XLEN-1:0] box_h_qnan = {{(`VX_CFG_XLEN-16){1'b1}}, qnan_h16};
        wire [`VX_CFG_XLEN-1:0] box_h_inf  = {{(`VX_CFG_XLEN-16){1'b1}}, inf_h16};
        wire [`VX_CFG_XLEN-1:0] box_h_zero = {{(`VX_CFG_XLEN-16){1'b1}}, zero_h16};
        wire [`VX_CFG_XLEN-1:0] box_h_max  = {{(`VX_CFG_XLEN-16){1'b1}}, max_h16};

        wire [`VX_CFG_XLEN-1:0] normal_dst = dst_d_s5 ? fp64_boxed : dst_h_s5 ? box_h_res : f32_boxed;

        wire is_nan_src  = is_f2f_s5 & fclass_s5.is_nan;
        wire is_inf_src  = is_f2f_s5 & fclass_s5.is_inf;
        wire is_zero_src = is_f2f_s5 & fclass_s5.is_zero;
        wire is_fin_src  = is_f2f_s5 & ~fclass_s5.is_nan & ~fclass_s5.is_inf & ~fclass_s5.is_zero;

        wire of_to_inf = (frm_s5 == INST_FRM_RNE) || (frm_s5 == INST_FRM_RMM)
                      || ((frm_s5 == INST_FRM_RUP) && ~input_sign_s5)
                      || ((frm_s5 == INST_FRM_RDN) &&  input_sign_s5);
        wire of_post   = is_fin_src & ~dst_d_s5 & ~f2f_of_s5
                       & (dst_h_s5 ? (fp16_exp == 5'h1F) : (fp32_exp == 8'hFF));

        wire [1:0] fine_rounded;
        VX_fp_rounding #(
            .DAT_WIDTH (2)
        ) fine_rounding (
            .abs_value_i         (2'b01),
            .sign_i              (input_sign_s5),
            .round_sticky_bits_i (fine_round_s5[1:0]),
            .rnd_mode_i          (frm_s5),
            .effective_subtraction_i (1'b0),
            .abs_rounded_o       (fine_rounded),
            `UNUSED_PIN          (sign_o),
            `UNUSED_PIN          (exact_zero_o),
            `UNUSED_PIN          (round_up_o)
        );
        `UNUSED_VAR (fine_rounded[0])
        wire not_tiny = f2f_tiny0_s5 & fine_round_s5[2] & fine_rounded[1];

        reg [`VX_CFG_XLEN-1:0] dst_r;
        always_comb begin
            if (is_nan_src) begin
                dst_r = dst_d_s5 ? qnan_d : dst_h_s5 ? box_h_qnan : box_s_qnan;
            end else if (is_inf_src) begin
                dst_r = dst_d_s5 ? inf_d : dst_h_s5 ? box_h_inf : box_s_inf;
            end else if (is_zero_src) begin
                dst_r = dst_d_s5 ? zero_d : dst_h_s5 ? box_h_zero : box_s_zero;
            end else if (f2f_of_s5) begin
                dst_r = dst_h_s5 ? (of_to_inf ? box_h_inf : box_h_max)
                                 : (of_to_inf ? box_s_inf : box_s_max);
            end else begin
                dst_r = normal_dst;
            end
        end
        assign fp_dst_res    = dst_r;
        assign f2f_nv_s5     = is_f2f_s5 & fclass_s5.is_signaling;
        assign f2f_of_out_s5 = f2f_of_s5 | of_post;
        assign f2f_uf_out_s5 = f2f_uf_s5 & ~not_tiny;
        assign f2f_nx_s5     = is_fin_src & (rs_any_s5 | f2f_of_out_s5);
    end else begin : g_fp_pack_s
        `UNUSED_VAR ({dst_h_s5, f2f_of_s5, f2f_tiny0_s5, fine_round_s5, frm_s5})
        assign fp_dst_res    = f32_boxed;
        assign f2f_nv_s5     = 1'b0;
        assign f2f_nx_s5     = 1'b0;
        assign f2f_of_out_s5 = 1'b0;
        assign f2f_uf_out_s5 = 1'b0;
    end

    // F2I saturation values for NaN/Inf inputs
    wire use_neg_sat_s5 = fclass_s5.is_inf && input_sign_s5;
    wire [31:0] nan_inf_32 = use_neg_sat_s5 ? (is_signed_s5 ? 32'h80000000 : 32'h00000000)
                                             : (is_signed_s5 ? 32'h7FFFFFFF : 32'hFFFFFFFF);
    wire [63:0] nan_inf_64 = use_neg_sat_s5 ? (is_signed_s5 ? 64'h8000000000000000 : 64'h0000000000000000)
                                             : (is_signed_s5 ? 64'h7FFFFFFFFFFFFFFF : 64'hFFFFFFFFFFFFFFFF);

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
                final_fflags_s5.NX = rs_any_s5;
            end
        end else begin
            // I2F or F2F: float destination
            res_val_64 = 64'(fp_dst_res);
            res_val_32 = fp_dst_res[31:0];
            final_fflags_s5.NV = f2f_nv_s5;
            final_fflags_s5.OF = f2f_of_out_s5;
            final_fflags_s5.UF = f2f_uf_out_s5 & rs_any_s5;
            final_fflags_s5.NX = is_f2f_s5 ? f2f_nx_s5 : rs_any_s5;
        end
    end

    `UNUSED_VAR ({dst_d_s5, fclass_s5, safe_rounded_abs, safe_dst_exp, res_val_64, res_val_32})

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
