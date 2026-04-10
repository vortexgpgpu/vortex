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

// Supports F32 (MAN_BITS=23, EXP_BITS=8) and F64 (MAN_BITS=52, EXP_BITS=11).
// Supports all RISC-V rounding modes and full fflags output.
// Minimum LATENCY is 6 (1 ini + 1 mul + 1 aln + 1 acc + 1 nrm + 1 rnd).

`include "VX_fpu_define.vh"

module VX_fma_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 6,
    parameter MAN_BITS = 23,  // mantissa bits (excluding hidden bit): 23=F32, 52=F64
    parameter EXP_BITS = 8    // exponent bits: 8=F32, 11=F64
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire mask,

    input  wire [INST_FPU_BITS-1:0] op_type,
    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [MAN_BITS+EXP_BITS:0] dataa,
    input  wire [MAN_BITS+EXP_BITS:0] datab,
    input  wire [MAN_BITS+EXP_BITS:0] datac,

    output wire [MAN_BITS+EXP_BITS:0] result,
    output wire [`FP_FLAGS_BITS-1:0]  fflags
);
    // =========================================================================
    // Latency parameters
    // =========================================================================
    localparam INI_LATENCY = 1;
    localparam ALN_LATENCY = 1;
    localparam ACC_LATENCY = 1;
    localparam NRM_LATENCY = 1;
    localparam RND_LATENCY = 1;
    localparam MUL_LATENCY = LATENCY - INI_LATENCY - ALN_LATENCY - ACC_LATENCY - NRM_LATENCY - RND_LATENCY;
    `STATIC_ASSERT(MUL_LATENCY >= 1, ("LATENCY must be >= %0d, got %0d", INI_LATENCY+1+ALN_LATENCY+ACC_LATENCY+NRM_LATENCY+RND_LATENCY, LATENCY))

    // Mask pipeline: tracks valid bits through every cycle
    reg [LATENCY-1:0] mask_pipe;
    always @(posedge clk) begin
        if (reset) begin
            mask_pipe <= '0;
        end else if (enable) begin
            mask_pipe <= {mask_pipe[LATENCY-2:0], mask};
        end
    end
    wire valid_aln = mask_pipe[INI_LATENCY+MUL_LATENCY-1];
    wire valid_acc = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY-1];
    wire valid_nrm = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY+ACC_LATENCY-1];
    wire valid_rnd = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY+ACC_LATENCY+NRM_LATENCY-1];

    // =========================================================================
    // FP field widths
    // =========================================================================
    localparam FLOAT_BITS  = 1 + EXP_BITS + MAN_BITS;
    localparam EXP_IWIDTH  = EXP_BITS + 2;
    localparam EXP_BIAS    = (1 << (EXP_BITS - 1)) - 1;
    localparam EXP_MAX     = (1 << EXP_BITS) - 1;
    localparam SIG_BITS    = MAN_BITS + 1;
    localparam PROD_BITS   = 2 * SIG_BITS;
    localparam ALN_BITS    = PROD_BITS + 3;
    localparam ACC_BITS    = ALN_BITS + 1;
    localparam LZC_BITS    = `LOG2UP(ACC_BITS);
    localparam NORM_WIN_BITS = MAN_BITS + 4;

    // =========================================================================
    // Stage 0 (INIT): decode, classify, form canonical operands
    //   No subnormal normalization — raw mantissa passed through.
    //   Extra leading zeros in subnormal significands are absorbed by
    //   the normalization LZC in the NORM stage; exponents compensate.
    // =========================================================================

    `UNUSED_VAR (fmt[0])
    wire is_sub   = fmt[1];
    wire is_nmadd = (op_type == INST_FPU_NMADD);
    wire is_mul   = (op_type == INST_FPU_MUL);
    wire is_add   = (op_type == INST_FPU_ADD);

    wire s_prod_neg = is_nmadd;
    wire s_c_neg    = is_sub ^ is_nmadd;

    wire [FLOAT_BITS-1:0] op_a = dataa;
    wire [FLOAT_BITS-1:0] op_b = is_add ? {1'b0, EXP_BITS'(EXP_BIAS), {MAN_BITS{1'b0}}} : datab;
    wire [FLOAT_BITS-1:0] op_c = is_add ? datab : datac;

    wire        s_a0 = op_a[FLOAT_BITS-1] ^ s_prod_neg;
    wire        s_b0 = op_b[FLOAT_BITS-1];
    wire        s_c0 = is_mul ? 1'b0 : (op_c[FLOAT_BITS-1] ^ s_c_neg);

    wire [EXP_BITS-1:0] e_a0 = op_a[FLOAT_BITS-2:MAN_BITS];
    wire [EXP_BITS-1:0] e_b0 = op_b[FLOAT_BITS-2:MAN_BITS];
    wire [EXP_BITS-1:0] e_c0 = is_mul ? {EXP_BITS{1'b0}} : op_c[FLOAT_BITS-2:MAN_BITS];

    wire [MAN_BITS-1:0] m_a0 = op_a[MAN_BITS-1:0];
    wire [MAN_BITS-1:0] m_b0 = op_b[MAN_BITS-1:0];
    wire [MAN_BITS-1:0] m_c0 = is_mul ? {MAN_BITS{1'b0}} : op_c[MAN_BITS-1:0];

    // Classify canonical operands
    fclass_t clss_a0, clss_b0, clss_c0;
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_a0 (
        .exp_i(e_a0), .man_i(m_a0), .clss_o(clss_a0));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_b0 (
        .exp_i(e_b0), .man_i(m_b0), .clss_o(clss_b0));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_c0 (
        .exp_i(e_c0), .man_i(m_c0), .clss_o(clss_c0));
    `UNUSED_VAR ({clss_a0.is_quiet,
                  clss_b0.is_quiet,
                  clss_c0.is_zero,
                  clss_c0.is_quiet})

    // Significands: hidden bit = 1 only for normals, 0 for zero/subnormal/inf/nan
    // No LZC or barrel shift — subnormal mantissa passed raw
    wire [SIG_BITS-1:0] sig_a = {clss_a0.is_normal, m_a0};
    wire [SIG_BITS-1:0] sig_b = {clss_b0.is_normal, m_b0};
    wire [SIG_BITS-1:0] sig_c = {clss_c0.is_normal, m_c0};

    // Biased exponents — no subnormal LZC adjustment
    // IEEE 754: subnormal has biased exponent 0 but true exponent = 1 - bias,
    // so we use biased exponent 1 for subnormals (matching the hidden-bit=0 convention).
    // For zero/inf/nan the exponent value doesn't matter (handled by exception path).
    wire signed [EXP_IWIDTH-1:0] exp_a = clss_a0.is_subnormal ? EXP_IWIDTH'(1)
                                        : clss_a0.is_normal    ? EXP_IWIDTH'(e_a0)
                                        :                        EXP_IWIDTH'(0);
    wire signed [EXP_IWIDTH-1:0] exp_b = clss_b0.is_subnormal ? EXP_IWIDTH'(1)
                                        : clss_b0.is_normal    ? EXP_IWIDTH'(e_b0)
                                        :                        EXP_IWIDTH'(0);
    wire signed [EXP_IWIDTH-1:0] exp_c = clss_c0.is_subnormal ? EXP_IWIDTH'(1)
                                        : clss_c0.is_normal    ? EXP_IWIDTH'(e_c0)
                                        :                        EXP_IWIDTH'(0);

    // Product sign and exponent
    wire s_prod0 = s_a0 ^ s_b0;
    wire signed [EXP_IWIDTH-1:0] exp_prod0 = (clss_a0.is_zero | clss_b0.is_zero) ? EXP_IWIDTH'(0)
                                : exp_a + exp_b - $signed(EXP_IWIDTH'(EXP_BIAS));

    // Early exception detection
    wire inf_a  = clss_a0.is_inf;
    wire inf_b  = clss_b0.is_inf;
    wire inf_c  = clss_c0.is_inf;
    wire nan_a  = clss_a0.is_nan;
    wire nan_b  = clss_b0.is_nan;
    wire nan_c  = clss_c0.is_nan;
    wire snan_a = clss_a0.is_signaling;
    wire snan_b = clss_b0.is_signaling;
    wire snan_c = clss_c0.is_signaling;

    wire nv_inf_zero = (inf_a & clss_b0.is_zero) | (clss_a0.is_zero & inf_b);
    wire nv_snan     = snan_a | snan_b | snan_c;
    wire prod_is_inf = (inf_a | inf_b) & ~nan_a & ~nan_b;
    wire nv_inf_inf  = prod_is_inf & inf_c & (s_prod0 != s_c0);

    wire early_nv    = nv_snan | nv_inf_zero | nv_inf_inf;
    wire any_nan     = nan_a | nan_b | nan_c;
    wire result_nan  = any_nan | early_nv;
    wire result_inf  = (prod_is_inf | inf_c) & ~result_nan & ~nv_inf_inf;
    wire result_inf_sign = prod_is_inf ? s_prod0 : s_c0;

    // [3]=result_nan [2]=result_inf [1]=result_inf_sign [0]=early_nv
    wire [3:0] exc0 = {result_nan, result_inf, result_inf_sign, early_nv};

    // =========================================================================
    // INIT → MUL register
    // =========================================================================
    localparam INI_DATAW = 3*SIG_BITS + EXP_IWIDTH + EXP_IWIDTH + 1 + 1 + INST_FRM_BITS + 4;

    wire [INI_DATAW-1:0] s0_data;
    VX_pipe_register #(
        .DATAW (INI_DATAW),
        .DEPTH (INI_LATENCY)
    ) pipe_ini (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && mask),
        .data_in ({sig_a, sig_b, sig_c, exp_prod0, exp_c, s_prod0, s_c0, frm, exc0}),
        .data_out(s0_data)
    );

    wire [SIG_BITS-1:0]           r1_sig_a, r1_sig_b, r1_sig_c;
    wire signed [EXP_IWIDTH-1:0]  r1_exp_prod, r1_exp_c;
    wire                          r1_s_prod, r1_s_c;
    wire [INST_FRM_BITS-1:0]      r1_frm;
    wire [3:0]                    r1_exc;
    assign {r1_sig_a, r1_sig_b, r1_sig_c, r1_exp_prod, r1_exp_c, r1_s_prod, r1_s_c, r1_frm, r1_exc} = s0_data;

    // =========================================================================
    // Stage 1 (MUL): MUL_LATENCY cycles — inferred multiply
    // =========================================================================

    wire [PROD_BITS-1:0] raw_prod;

    if (MUL_LATENCY < `LATENCY_IMUL && SIG_BITS <= 24) begin : g_mul_wallace
        VX_wallace_mul #(
            .N     (SIG_BITS),
            .P     (PROD_BITS),
            .CPA_KS(!`FORCE_BUILTIN_ADDER(PROD_BITS))
        ) u_mul (
            .a (r1_sig_a),
            .b (r1_sig_b),
            .p (raw_prod)
        );
    end else begin : g_mul_infer
        assign raw_prod = PROD_BITS'(r1_sig_a) * PROD_BITS'(r1_sig_b);
    end

    localparam MUL_DATAW = PROD_BITS + EXP_IWIDTH + 1 + SIG_BITS + EXP_IWIDTH + 1 + INST_FRM_BITS + 4;

    wire [MUL_DATAW-1:0] s1_data;
    VX_pipe_register #(
        .DATAW (MUL_DATAW),
        .DEPTH (MUL_LATENCY)
    ) pipe_mul (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({raw_prod, r1_exp_prod, r1_s_prod, r1_sig_c, r1_exp_c, r1_s_c, r1_frm, r1_exc}),
        .data_out(s1_data)
    );

    wire [PROD_BITS-1:0]          s1_prod;
    wire signed [EXP_IWIDTH-1:0]  s1_exp_prod;
    wire                          s1_s_prod;
    wire [SIG_BITS-1:0]           s1_sig_c;
    wire signed [EXP_IWIDTH-1:0]  s1_exp_c;
    wire                          s1_s_c;
    wire [INST_FRM_BITS-1:0]      s1_frm;
    wire [3:0]                    s1_exc;
    assign {s1_prod, s1_exp_prod, s1_s_prod, s1_sig_c, s1_exp_c, s1_s_c, s1_frm, s1_exc} = s1_data;

    // =========================================================================
    // Stage 2 (ALIGN): align product and C addend
    //   Single barrel shifter — only the smaller operand is shifted.
    //   Magnitude compare moved to ACC stage.
    // =========================================================================

    wire s1_eff_sub = s1_s_prod ^ s1_s_c;

    // Exponent comparison and shift amount
    wire prod_ge_c = (s1_exp_prod >= s1_exp_c);
    wire signed [EXP_IWIDTH-1:0] s1_max_exp = prod_ge_c ? s1_exp_prod : s1_exp_c;

    localparam SHIFT_BITS = `LOG2UP(ALN_BITS + 1);
    wire signed [EXP_IWIDTH-1:0] exp_diff = prod_ge_c ? (s1_exp_prod - s1_exp_c)
                                                       : (s1_exp_c - s1_exp_prod);
    wire [SHIFT_BITS-1:0] shift_amt = (exp_diff > $signed(EXP_IWIDTH'(ALN_BITS)))
                                    ? SHIFT_BITS'(ALN_BITS)
                                    : SHIFT_BITS'(exp_diff);

    // Extend product into ALN_BITS-wide field with 3 GRS guard bits at bottom
    wire [ALN_BITS-1:0] prod_aligned_full = {s1_prod, {(ALN_BITS-PROD_BITS){1'b0}}};

    // Place C significand aligned with product: implicit-1 at bit (ALN_BITS-2)
    wire [ALN_BITS-1:0] c_aligned_full = {1'b0, s1_sig_c, {(ALN_BITS-SIG_BITS-1){1'b0}}};

    // Shift the smaller operand right; collect sticky from shifted-out bits
    wire [ALN_BITS-1:0] shift_in  = prod_ge_c ? c_aligned_full : prod_aligned_full;
    wire [ALN_BITS-1:0] shift_out;

    // Variable right-shift with sticky collection
    wire [ALN_BITS+ALN_BITS-1:0] shift_ext = {shift_in, {ALN_BITS{1'b0}}};
    wire [ALN_BITS+ALN_BITS-1:0] shifted_ext = shift_ext >> shift_amt;
    assign shift_out = shifted_ext[ALN_BITS+ALN_BITS-1 : ALN_BITS];
    wire shift_sticky = |shifted_ext[ALN_BITS-1:0];

    // Assign aligned operands
    wire [ALN_BITS-1:0] aln_prod = prod_ge_c ? prod_aligned_full : shift_out;
    wire [ALN_BITS-1:0] aln_c    = prod_ge_c ? shift_out : c_aligned_full;
    wire aln_sticky = shift_sticky;

    // =========================================================================
    // ALIGN → ACC register
    // =========================================================================
    localparam ALN_DATAW = ALN_BITS + ALN_BITS + 1 + 1 + 1 + 1 + EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [ALN_DATAW-1:0] s2_data;
    VX_pipe_register #(
        .DATAW (ALN_DATAW),
        .DEPTH (ALN_LATENCY)
    ) pipe_aln (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_aln),
        .data_in ({aln_prod, aln_c, aln_sticky, s1_eff_sub, s1_s_prod, s1_s_c, s1_max_exp, s1_frm, s1_exc}),
        .data_out(s2_data)
    );

    wire [ALN_BITS-1:0]           s2_aln_prod;
    wire [ALN_BITS-1:0]           s2_aln_c;
    wire                          s2_sticky;
    wire                          s2_eff_sub;
    wire                          s2_s_prod;
    wire                          s2_s_c;
    wire signed [EXP_IWIDTH-1:0]  s2_max_exp;
    wire [INST_FRM_BITS-1:0]      s2_frm;
    wire [3:0]                    s2_exc;
    assign {s2_aln_prod, s2_aln_c, s2_sticky, s2_eff_sub, s2_s_prod, s2_s_c, s2_max_exp, s2_frm, s2_exc} = s2_data;

    // =========================================================================
    // Stage 3 (ACC): accumulate + LZC
    //   Magnitude compare done here (moved from ALIGN).
    //   Exact LZC on acc_sum provides shift count for NORM stage.
    // =========================================================================

    // Magnitude comparison
    wire prod_gte_c = (s2_aln_prod >= s2_aln_c);

    // Addition and subtraction paths
    wire [ACC_BITS-1:0] add_result = {1'b0, s2_aln_prod} + {1'b0, s2_aln_c};
    wire [ACC_BITS-1:0] sub_ab     = {1'b0, s2_aln_prod} - {1'b0, s2_aln_c};

    // For subtraction with |C| > |prod|: negate (serial, OK — LZA is slower)
    wire [ACC_BITS-1:0] acc_sum;
    wire                acc_sign;

    assign acc_sum  = s2_eff_sub ? (prod_gte_c ? sub_ab : (~sub_ab + ACC_BITS'(1)))
                                : add_result;
    assign acc_sign = s2_eff_sub ? (prod_gte_c ? s2_s_prod : s2_s_c) : s2_s_prod;

    // --- Leading zero count on accumulated result ---
    // Exact LZC on acc_sum provides the shift count for normalization.
    // barrel shift and rounding are in separate stages (NORM, ROUND).
    wire [LZC_BITS-1:0] lzc_count;
    wire                lzc_valid;
    VX_lzc #(.N(ACC_BITS)) lzc_inst (
        .data_in  (acc_sum),
        .data_out (lzc_count),
        .valid_out(lzc_valid)
    );

    wire [LZC_BITS-1:0] lzc_predict = lzc_valid ? lzc_count : LZC_BITS'(ACC_BITS);

    wire acc_sticky = s2_sticky;

    // =========================================================================
    // ACC → NORM register
    // =========================================================================
    localparam ACC_DATAW = ACC_BITS + 1 + 1 + 1 + LZC_BITS + EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [ACC_DATAW-1:0] s3_data;
    VX_pipe_register #(
        .DATAW (ACC_DATAW),
        .DEPTH (ACC_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_acc),
        .data_in ({acc_sum, acc_sign, acc_sticky, s2_eff_sub, lzc_predict, s2_max_exp, s2_frm, s2_exc}),
        .data_out(s3_data)
    );

    wire [ACC_BITS-1:0]           s3_sum;
    wire                          s3_sign;
    wire                          s3_sticky;
    wire                          s3_eff_sub;
    wire [LZC_BITS-1:0]          s3_lzc_pred;
    wire signed [EXP_IWIDTH-1:0]  s3_max_exp;
    wire [INST_FRM_BITS-1:0]      s3_frm;
    wire [3:0]                    s3_exc;
    assign {s3_sum, s3_sign, s3_sticky, s3_eff_sub, s3_lzc_pred, s3_max_exp, s3_frm, s3_exc} = s3_data;

    // =========================================================================
    // Stage 4 (NORM): normalize using registered LZA prediction
    //   Barrel shift only — LZC was hoisted into ACC via LZA.
    //   Speculative exponents computed in parallel with the shift.
    // =========================================================================

    wire zero_sum = ~|s3_sum;

    // Normalize: shift left by predicted LZC
    wire [ACC_BITS:0] sum_ext     = {1'b0, s3_sum};
    wire [ACC_BITS:0] shifted_raw = sum_ext << s3_lzc_pred;
    wire              overshift   = shifted_raw[ACC_BITS];

    // Extract NORM_WIN_BITS-wide window: {hidden, MAN_BITS mantissa, G, R, S}
    wire [NORM_WIN_BITS-1:0] norm_window = overshift ? shifted_raw[ACC_BITS   -: NORM_WIN_BITS]
                                                     : shifted_raw[ACC_BITS-1 -: NORM_WIN_BITS];

    // Speculative exponents (parallel with barrel shift)
    // Reference: product implicit-1 at bit (ALN_BITS-2) when aligned, biased exp = max_exp.
    // After accumulation, leading one at bit (ACC_BITS-1-lzc) of acc_sum.
    // result_exp = max_exp - lzc_pred + 2 (base), +1 for overshift, +1 for round carry.
    localparam EXP_ADJ = 2;
    wire signed [EXP_IWIDTH-1:0] nrm_exp_base  = s3_max_exp - $signed(EXP_IWIDTH'(s3_lzc_pred)) + $signed(EXP_IWIDTH'(EXP_ADJ));
    wire signed [EXP_IWIDTH-1:0] nrm_exp_plus1 = nrm_exp_base + $signed(EXP_IWIDTH'(1));
    wire signed [EXP_IWIDTH-1:0] nrm_exp_plus2 = nrm_exp_base + $signed(EXP_IWIDTH'(2));

    // Sticky bits below the normalization window
    localparam STICK_IDX = MAN_BITS + 2;
    wire sticky_below = overshift ? (|shifted_raw[STICK_IDX:0])
                                  : (|shifted_raw[STICK_IDX-1:0]);

    // =========================================================================
    // NORM → ROUND register
    // =========================================================================
    localparam NRM_DATAW = NORM_WIN_BITS + 1 + 1 + 1 + 1 + 1 + 1
                         + 3*EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [NRM_DATAW-1:0] s4_data;
    VX_pipe_register #(
        .DATAW (NRM_DATAW),
        .DEPTH (NRM_LATENCY)
    ) pipe_nrm (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_nrm),
        .data_in ({norm_window, overshift, sticky_below, s3_sticky, s3_sign, s3_eff_sub, zero_sum,
                   nrm_exp_base, nrm_exp_plus1, nrm_exp_plus2, s3_frm, s3_exc}),
        .data_out(s4_data)
    );

    wire [NORM_WIN_BITS-1:0]      r5_window;
    wire                          r5_overshift;
    wire                          r5_sticky_below;
    wire                          r5_sticky_acc;
    wire                          r5_sign;
    wire                          r5_eff_sub;
    wire                          r5_zero_sum;
    wire signed [EXP_IWIDTH-1:0]  r5_exp_base, r5_exp_plus1, r5_exp_plus2;
    wire [INST_FRM_BITS-1:0]      r5_frm;
    wire [3:0]                    r5_exc;
    assign {r5_window, r5_overshift, r5_sticky_below, r5_sticky_acc, r5_sign, r5_eff_sub, r5_zero_sum,
            r5_exp_base, r5_exp_plus1, r5_exp_plus2, r5_frm, r5_exc} = s4_data;

    // =========================================================================
    // Stage 5 (ROUND): select-add rounding + result packing
    //   man+1 computed in parallel with round decision.
    //   Exponent selected by {overshift, round_carry} from pre-computed variants.
    // =========================================================================

    // Extract mantissa and rounding bits from registered window
    wire [MAN_BITS:0] rnd_man    = r5_window[NORM_WIN_BITS-1:3];
    wire              guard_bit  = r5_window[2];
    wire              round_bit  = r5_window[1];
    wire              sticky_sum = r5_window[0] | r5_sticky_below | r5_sticky_acc;

    // --- Select-add rounding: compute man+0 and man+1 in parallel ---
    wire [MAN_BITS:0] man_inc = rnd_man + (MAN_BITS+1)'(1);

    // Round decision (parallel with man+1 — depends only on GRS + frm + sign)
    reg round_up;
    wire [1:0] round_sticky_bits = {guard_bit, round_bit | sticky_sum};
    always @(*) begin
        case (r5_frm)
            INST_FRM_RNE:
                case (round_sticky_bits)
                    2'b00,
                    2'b01: round_up = 1'b0;
                    2'b10: round_up = rnd_man[0]; // tie to even
                    2'b11: round_up = 1'b1;
                endcase
            INST_FRM_RTZ: round_up = 1'b0;
            INST_FRM_RDN: round_up = (|round_sticky_bits) & r5_sign;
            INST_FRM_RUP: round_up = (|round_sticky_bits) & ~r5_sign;
            INST_FRM_RMM: round_up = round_sticky_bits[1];
            default:  round_up = 1'bx;
        endcase
    end

    // Select rounded mantissa
    wire [MAN_BITS:0] abs_rounded = round_up ? man_inc : rnd_man;

    // Round carry: mantissa overflowed (e.g. all-1s + 1 → 0)
    wire round_carry = round_up & (&rnd_man);

    wire [MAN_BITS-1:0] final_man = round_carry ? abs_rounded[MAN_BITS:1] : abs_rounded[MAN_BITS-1:0];

    // Exact zero and sign handling
    wire exact_zero = (rnd_man == '0) && (round_sticky_bits == '0);
    wire round_sign = (exact_zero && r5_eff_sub) ? (r5_frm == INST_FRM_RDN)
                                                  : r5_sign;

    // Exponent selection by {overshift, round_carry}
    logic signed [EXP_IWIDTH-1:0] final_exp_s;
    always_comb begin
        case ({r5_overshift, round_carry})
            2'b00: final_exp_s = r5_exp_base;
            2'b01: final_exp_s = r5_exp_plus1;
            2'b10: final_exp_s = r5_exp_plus1;
            2'b11: final_exp_s = r5_exp_plus2;
        endcase
    end

    // Exception flags
    wire is_nan_result   = r5_exc[3];
    wire is_inf_result   = r5_exc[2];
    wire inf_sign_result = r5_exc[1];
    wire nv_flag         = r5_exc[0];

    wire of_flag = (final_exp_s >= $signed(EXP_IWIDTH'(EXP_MAX))) & ~is_nan_result & ~is_inf_result;
    wire uf_flag = (final_exp_s <= $signed(EXP_IWIDTH'(0)))        & ~is_nan_result & ~is_inf_result & ~r5_zero_sum & ~exact_zero;
    wire nx_flag = (guard_bit | round_bit | sticky_sum) & ~is_nan_result & ~is_inf_result;

    // Final result mux
    logic [FLOAT_BITS-1:0] rnd_result;
    always_comb begin
        if (is_nan_result) begin
            rnd_result = {1'b0, {EXP_BITS{1'b1}}, 1'b1, {(MAN_BITS-1){1'b0}}};
        end else if (is_inf_result) begin
            rnd_result = {inf_sign_result, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};
        end else if (of_flag) begin
            rnd_result = {round_sign, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};
        end else if (r5_zero_sum | exact_zero | uf_flag) begin
            rnd_result = {round_sign, {(FLOAT_BITS-1){1'b0}}};
        end else begin
            rnd_result = {round_sign, final_exp_s[EXP_BITS-1:0], final_man};
        end
    end

    fflags_t rnd_fflags;
    assign rnd_fflags.NV = nv_flag;
    assign rnd_fflags.DZ = 1'b0;
    assign rnd_fflags.OF = of_flag;
    assign rnd_fflags.UF = uf_flag;
    assign rnd_fflags.NX = nx_flag | of_flag | uf_flag;

    // =========================================================================
    // ROUND output register
    // =========================================================================
    VX_pipe_register #(
        .DATAW (FLOAT_BITS + `FP_FLAGS_BITS),
        .DEPTH (RND_LATENCY)
    ) pipe_rnd (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_rnd),
        .data_in ({rnd_result, rnd_fflags}),
        .data_out({result,     fflags})
    );

endmodule
