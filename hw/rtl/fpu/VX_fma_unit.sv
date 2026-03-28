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

// Single-lane FP FMA pipeline: Multiply → Align → Accumulate → Normalize/Round
// Supports F32 (MAN_BITS=23, EXP_BITS=8) and F64 (MAN_BITS=52, EXP_BITS=11).
// Supports all RISC-V rounding modes and full fflags output.
// MUL_LATENCY is inferred from LATENCY: MUL_LATENCY = LATENCY - 3.
// When MUL_LATENCY < LATENCY_IMUL, a combinational Wallace tree is used;
// otherwise an inferred pipelined multiply is used (maps to DSP slices on FPGA).
// Minimum LATENCY is 5 (1 ini + 1 mul + 1 aln + 1 acc + 1 nrm).

`include "VX_fpu_define.vh"

module VX_fma_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 5,
    parameter MAN_BITS = 23,  // mantissa bits (excluding hidden bit): 23=F32, 52=F64
    parameter EXP_BITS = 8    // exponent bits: 8=F32, 11=F64
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,

    input  wire [INST_FPU_BITS-1:0] op_type,
    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [MAN_BITS+EXP_BITS:0] dataa,
    input  wire [MAN_BITS+EXP_BITS:0] datab,
    input  wire [MAN_BITS+EXP_BITS:0] datac,

    output wire [MAN_BITS+EXP_BITS:0]   result,
    output wire [`FP_FLAGS_BITS-1:0]    fflags
);
    // =========================================================================
    // Latency parameters
    // =========================================================================
    localparam INI_LATENCY = 1;
    localparam ALN_LATENCY = 1;
    localparam ACC_LATENCY = 1;
    localparam NRM_LATENCY = 1;
    localparam MUL_LATENCY = LATENCY - INI_LATENCY - ALN_LATENCY - ACC_LATENCY - NRM_LATENCY;
    `STATIC_ASSERT(MUL_LATENCY >= 1, ("LATENCY must be >= %0d, got %0d", INI_LATENCY+1+ALN_LATENCY+ACC_LATENCY+NRM_LATENCY, LATENCY))

    // =========================================================================
    // FP field widths
    // =========================================================================
    localparam FLOAT_BITS  = 1 + EXP_BITS + MAN_BITS;  // total bits: 32 (F32) or 64 (F64)
    localparam EXP_IWIDTH  = EXP_BITS + 2;              // signed intermediate exponent width
    localparam EXP_BIAS    = (1 << (EXP_BITS - 1)) - 1; // 127 (F32) or 1023 (F64)
    localparam EXP_MAX     = (1 << EXP_BITS) - 1;       // 255 (F32) or 2047 (F64)
    localparam SIG_BITS    = MAN_BITS + 1;               // with hidden bit: 24 or 53
    localparam PROD_BITS   = 2 * SIG_BITS;               // 48 or 106
    // Aligned accumulator width: product + 3 guard/round/sticky bits
    localparam ALN_BITS    = PROD_BITS + 3;              // 51 or 109
    // Accumulator output: one extra carry bit
    localparam ACC_BITS    = ALN_BITS + 1;               // 52 or 110
    // LZC widths
    localparam SUBLZC_BITS = `LOG2UP(SIG_BITS);          // 5 (F32) or 6 (F64)
    localparam LZC_BITS    = `LOG2UP(ACC_BITS);          // 6 (F32) or 7 (F64)
    // Normalization window: {hidden, MAN_BITS mantissa, G, R, S} = MAN_BITS+4 bits
    localparam NORM_WIN_BITS = MAN_BITS + 4;             // 27 (F32) or 56 (F64)
    // Sticky bit index below normalization window
    localparam STICK_IDX   = MAN_BITS + 2;              // ACC_BITS - NORM_WIN_BITS = MAN_BITS+2

    // =========================================================================
    // Stage 0 (combinational): decode, classify, form canonical A/B/C
    // =========================================================================

    // --- Decode op → effective sign flips and C source ---
    // op_type: INST_FPU_ADD=0000, INST_FPU_MUL=0001, INST_FPU_MADD=0010, INST_FPU_NMADD=0011
    // fmt[1]: is_sub flag (FSUB / FMSUB / FNMSUB); fmt[0] = S/D format (unused here)
    `UNUSED_VAR (fmt[0])
    wire is_sub  = fmt[1];
    wire is_nmadd= (op_type == INST_FPU_NMADD);
    wire is_mul  = (op_type == INST_FPU_MUL);
    wire is_add  = (op_type == INST_FPU_ADD);

    // Effective sign of product: negate A for FNMADD/FNMSUB
    wire s_prod_neg = is_nmadd;

    // Effective sign of C addend
    // FMADD:  +c   FMSUB:  -c   FNMADD: -c   FNMSUB: +c   FADD: +c  FSUB: -c  FMUL: C=0
    wire s_c_neg = is_sub ^ is_nmadd;

    // Canonical A/B significands (implicit 1, subnormals handled later)
    // For ADD/SUB: A=dataa, B=1.0, C=datab  (sign flip on C for SUB)
    // For MUL:     C forced to +0
    wire [FLOAT_BITS-1:0] op_a = dataa;
    wire [FLOAT_BITS-1:0] op_b = is_add ? {1'b0, EXP_BITS'(EXP_BIAS), {MAN_BITS{1'b0}}} : datab;
    wire [FLOAT_BITS-1:0] op_c = is_add ? datab : datac;

    wire        s_a0 = op_a[FLOAT_BITS-1] ^ s_prod_neg;
    wire        s_b0 = op_b[FLOAT_BITS-1];
    wire        s_c0 = (is_mul) ? 1'b0 : (op_c[FLOAT_BITS-1] ^ s_c_neg);

    wire [EXP_BITS-1:0] e_a0 = op_a[FLOAT_BITS-2:MAN_BITS];
    wire [EXP_BITS-1:0] e_b0 = op_b[FLOAT_BITS-2:MAN_BITS];
    wire [EXP_BITS-1:0] e_c0 = is_mul ? {EXP_BITS{1'b0}} : op_c[FLOAT_BITS-2:MAN_BITS];

    wire [MAN_BITS-1:0] m_a0 = op_a[MAN_BITS-1:0];
    wire [MAN_BITS-1:0] m_b0 = op_b[MAN_BITS-1:0];
    wire [MAN_BITS-1:0] m_c0 = is_mul ? {MAN_BITS{1'b0}} : op_c[MAN_BITS-1:0];

    // Re-classify with canonical operands
    fclass_t clss_a0, clss_b0, clss_c0;
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_a0 (
        .exp_i(e_a0), .man_i(m_a0), .clss_o(clss_a0));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_b0 (
        .exp_i(e_b0), .man_i(m_b0), .clss_o(clss_b0));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_c0 (
        .exp_i(e_c0), .man_i(m_c0), .clss_o(clss_c0));
    // is_normal and is_quiet not needed by FMA logic
    `UNUSED_VAR ({clss_a0.is_normal, clss_a0.is_quiet,
                  clss_b0.is_normal, clss_b0.is_quiet,
                  clss_c0.is_normal, clss_c0.is_quiet})

    // --- Subnormal normalization for significands ---
    // For subnormals: shift mantissa left until MSB=1, reduce exponent accordingly.
    // We use a simple leading-zero shift here (combinational, pre-MUL stage).
    wire [SIG_BITS-1:0] sig_a_raw = {~clss_a0.is_zero & ~clss_a0.is_subnormal ? 1'b1 : 1'b0, m_a0};
    wire [SIG_BITS-1:0] sig_b_raw = {~clss_b0.is_zero & ~clss_b0.is_subnormal ? 1'b1 : 1'b0, m_b0};
    wire [SIG_BITS-1:0] sig_c_raw = {~clss_c0.is_zero & ~clss_c0.is_subnormal ? 1'b1 : 1'b0, m_c0};

    // Subnormal shift counts
    wire [SUBLZC_BITS-1:0] lzc_a, lzc_b, lzc_c;
    wire lzc_a_vld, lzc_b_vld, lzc_c_vld;
    VX_lzc #(.N(SIG_BITS)) lzc_suba (.data_in(sig_a_raw), .data_out(lzc_a), .valid_out(lzc_a_vld));
    VX_lzc #(.N(SIG_BITS)) lzc_subb (.data_in(sig_b_raw), .data_out(lzc_b), .valid_out(lzc_b_vld));
    VX_lzc #(.N(SIG_BITS)) lzc_subc (.data_in(sig_c_raw), .data_out(lzc_c), .valid_out(lzc_c_vld));
    `UNUSED_VAR ({lzc_a_vld, lzc_b_vld, lzc_c_vld})

    // Normalized significands (shift subnormals left)
    wire [SIG_BITS-1:0] sig_a = clss_a0.is_subnormal ? (sig_a_raw << lzc_a) : sig_a_raw;
    wire [SIG_BITS-1:0] sig_b = clss_b0.is_subnormal ? (sig_b_raw << lzc_b) : sig_b_raw;
    wire [SIG_BITS-1:0] sig_c = clss_c0.is_subnormal ? (sig_c_raw << lzc_c) : sig_c_raw;

    // Biased exponents, adjusted for subnormal shift; use EXP_IWIDTH bits (signed) to
    // allow negative exponents during intermediate computation
    wire signed [EXP_IWIDTH-1:0] exp_a = clss_a0.is_subnormal ? ($signed(EXP_IWIDTH'(1)) - $signed(EXP_IWIDTH'(lzc_a)))
                                        : clss_a0.is_zero       ? EXP_IWIDTH'(0)
                                        :                         EXP_IWIDTH'(e_a0);
    wire signed [EXP_IWIDTH-1:0] exp_b = clss_b0.is_subnormal ? ($signed(EXP_IWIDTH'(1)) - $signed(EXP_IWIDTH'(lzc_b)))
                                        : clss_b0.is_zero       ? EXP_IWIDTH'(0)
                                        :                         EXP_IWIDTH'(e_b0);
    wire signed [EXP_IWIDTH-1:0] exp_c = clss_c0.is_subnormal ? ($signed(EXP_IWIDTH'(1)) - $signed(EXP_IWIDTH'(lzc_c)))
                                        : clss_c0.is_zero       ? EXP_IWIDTH'(0)
                                        :                         EXP_IWIDTH'(e_c0);

    // Product sign
    wire s_prod0 = s_a0 ^ s_b0;

    // Product biased exponent (before normalization): ea + eb - bias
    wire signed [EXP_IWIDTH-1:0] exp_prod0 = (clss_a0.is_zero | clss_b0.is_zero) ? EXP_IWIDTH'(0)
                                :  exp_a + exp_b - $signed(EXP_IWIDTH'(EXP_BIAS));

    // Early exception flags (passed through pipeline)
    // NV: SNaN input, or Inf*0, or Inf-Inf
    wire inf_a = clss_a0.is_inf;
    wire inf_b = clss_b0.is_inf;
    wire inf_c = clss_c0.is_inf;
    wire nan_a = clss_a0.is_nan;
    wire nan_b = clss_b0.is_nan;
    wire nan_c = clss_c0.is_nan;
    wire snan_a= clss_a0.is_signaling;
    wire snan_b= clss_b0.is_signaling;
    wire snan_c= clss_c0.is_signaling;
    wire zer_a = clss_a0.is_zero;
    wire zer_b = clss_b0.is_zero;

    wire nv_inf_zero  = (inf_a & zer_b) | (zer_a & inf_b);
    wire nv_snan      = snan_a | snan_b | snan_c;
    // Inf - Inf: product is Inf and C is Inf with opposite sign
    wire prod_is_inf  = (inf_a | inf_b) & ~nan_a & ~nan_b;
    wire nv_inf_inf   = prod_is_inf & inf_c & (s_prod0 != s_c0);

    wire early_nv = nv_snan | nv_inf_zero | nv_inf_inf;

    // Special-case result flags (propagate through pipeline)
    wire any_nan     = nan_a | nan_b | nan_c;
    wire result_nan  = any_nan | early_nv;
    // Result is Inf when: one input Inf (and no NaN, no cancellation)
    wire result_inf  = (prod_is_inf | inf_c) & ~result_nan & ~nv_inf_inf;
    wire result_inf_sign = prod_is_inf ? s_prod0 : s_c0;

    // Pack early flags into a small struct for pipeline transport
    // [3]=result_nan [2]=result_inf [1]=result_inf_sign [0]=early_nv
    wire [3:0] exc0 = {result_nan, result_inf, result_inf_sign, early_nv};

    // =========================================================================
    // Stage 0→1 register: break the classify→LZC→shift→multiply critical path
    // =========================================================================
    localparam PRE_MUL_DATAW = 3*SIG_BITS + EXP_IWIDTH + EXP_IWIDTH + 1 + 1 + INST_FRM_BITS + 4;

    wire [PRE_MUL_DATAW-1:0] s0_data;
    VX_pipe_register #(
        .DATAW (PRE_MUL_DATAW),
        .DEPTH (INI_LATENCY)
    ) pipe_ini (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
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
    // Stage 1: Multiply  (MUL_LATENCY cycles)
    // =========================================================================

    wire [PROD_BITS-1:0] raw_prod;

    // Use Wallace tree only for small significands (F32: SIG_BITS=24); for larger
    // widths (F64: SIG_BITS=53) the CSA tree produces unused upper bits, so fall
    // back to an inferred multiply which synthesis maps to DSP slices.
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

    // Pipeline register for MUL stage: MUL_LATENCY already excludes INI_LATENCY
    // Carry forward: raw_prod, exp_prod, s_prod, sig_c, exp_c, s_c, frm, exc
    localparam MUL_DATAW = PROD_BITS + EXP_IWIDTH + 1 + SIG_BITS + EXP_IWIDTH + 1 + INST_FRM_BITS + 4;

    wire [MUL_DATAW-1:0] s1_data;
    VX_pipe_register #(
        .DATAW (MUL_DATAW),
        .DEPTH (MUL_LATENCY)
    ) pipe_mul (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({raw_prod,  r1_exp_prod, r1_s_prod, r1_sig_c, r1_exp_c, r1_s_c, r1_frm, r1_exc}),
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
    // Stage 2: Align (1 cycle)
    // =========================================================================
    // Represent both operands in the same ALN_BITS fixed-point format.
    // Product is PROD_BITS wide → extend to ALN_BITS by appending 3 zero LSBs
    // C is SIG_BITS wide → extend to ALN_BITS by left-shifting to align with product MSB
    //
    // Strategy:
    //   max_exp = max(s1_exp_prod, s1_exp_c)
    //   product: shift right by (max_exp - s1_exp_prod)  [0 if product is larger]
    //   C:       shift right by (max_exp - s1_exp_c)     [0 if C is larger]
    // Each shift appends to a sticky accumulator.

    // Effective subtraction
    wire s1_eff_sub = s1_s_prod ^ s1_s_c;

    // Determine max exponent
    wire prod_ge_c = (s1_exp_prod >= s1_exp_c);
    wire signed [EXP_IWIDTH-1:0] s1_max_exp = prod_ge_c ? s1_exp_prod : s1_exp_c;

    // Shift amounts (clamped to ALN_BITS+1 to avoid wrap)
    localparam SHIFT_BITS = `LOG2UP(ALN_BITS + 1);
    wire [SHIFT_BITS-1:0] shift_prod = prod_ge_c ? '0
                        : (s1_exp_c - s1_exp_prod) > ALN_BITS ? SHIFT_BITS'(ALN_BITS)
                        :                                        SHIFT_BITS'(s1_exp_c - s1_exp_prod);
    wire [SHIFT_BITS-1:0] shift_c    = prod_ge_c
                        ? ((s1_exp_prod - s1_exp_c) > ALN_BITS ? SHIFT_BITS'(ALN_BITS)
                                                                : SHIFT_BITS'(s1_exp_prod - s1_exp_c))
                        : '0;

    // Extend and shift product. prod_ext is 2*ALN_BITS wide:
    // s1_prod (PROD_BITS) sits at the top, followed by (2*ALN_BITS-PROD_BITS) zeros.
    wire [ALN_BITS+ALN_BITS-1:0] prod_ext = {s1_prod, {(ALN_BITS+ALN_BITS-PROD_BITS){1'b0}}};
    wire [ALN_BITS-1:0] aln_prod  = ALN_BITS'(prod_ext >> (ALN_BITS + shift_prod));
    wire                stick_prod = (prod_ext << (ALN_BITS - shift_prod)) != '0;

    // Extend and shift C: place C's implicit-1 at bit (ALN_BITS-2), matching the
    // product's implicit-1 position (raw_prod bit[PROD_BITS-2] lands at aln_prod[ALN_BITS-2] at shift_prod=0).
    wire [ALN_BITS-1:0] c_left_aligned = {1'b0, s1_sig_c, {(ALN_BITS-SIG_BITS-1){1'b0}}};
    wire [ALN_BITS+ALN_BITS-1:0] c_ext = {c_left_aligned, {ALN_BITS{1'b0}}};
    wire [ALN_BITS-1:0] aln_c    = ALN_BITS'(c_ext >> (ALN_BITS + shift_c));
    wire                stick_c  = (c_ext << (ALN_BITS - shift_c)) != '0;

    // Pre-compute magnitude comparison in ALN stage (before register) to reduce
    // critical-path depth in the ACC stage.
    wire aln_prod_gte_c = (aln_prod >= aln_c);

    // Pipeline register for ALN stage
    localparam ALN_DATAW = ALN_BITS + 1 + ALN_BITS + 1 + 1 + 1 + 1 + 1 + EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [ALN_DATAW-1:0] s2_data;
    VX_pipe_register #(
        .DATAW (ALN_DATAW),
        .DEPTH (ALN_LATENCY)
    ) pipe_aln (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({aln_prod,  stick_prod, aln_c,  stick_c, aln_prod_gte_c, s1_eff_sub, s1_s_prod, s1_s_c, s1_max_exp, s1_frm, s1_exc}),
        .data_out(s2_data)
    );

    wire [ALN_BITS-1:0]           s2_aln_prod;
    wire                          s2_stick_prod;
    wire [ALN_BITS-1:0]           s2_aln_c;
    wire                          s2_stick_c;
    wire                          s2_prod_gte_c;
    wire                          s2_eff_sub;
    wire                          s2_s_prod;
    wire                          s2_s_c;
    wire signed [EXP_IWIDTH-1:0]  s2_max_exp;
    wire [INST_FRM_BITS-1:0]      s2_frm;
    wire [3:0]                    s2_exc;

    assign {s2_aln_prod, s2_stick_prod, s2_aln_c, s2_stick_c,
            s2_prod_gte_c, s2_eff_sub, s2_s_prod, s2_s_c, s2_max_exp, s2_frm, s2_exc} = s2_data;

    // =========================================================================
    // Stage 3: Accumulate (1 cycle)
    // =========================================================================
    // Both operands are positive magnitudes in sign-magnitude form.
    // Result sign is determined by the larger magnitude.
    // The magnitude comparison (prod >= c) was pre-computed and registered
    // in the ALN stage to shorten this stage's critical path.

    wire [ACC_BITS-1:0] acc_sum;
    wire                acc_sign;
    wire                acc_sticky;

    wire [ACC_BITS-1:0] add_result = {1'b0, s2_aln_prod} + {1'b0, s2_aln_c};
    wire [ACC_BITS-1:0] sub_ab     = {1'b0, s2_aln_prod} - {1'b0, s2_aln_c};

    assign acc_sum    = s2_eff_sub ? (s2_prod_gte_c ? sub_ab : (~sub_ab + 1'b1)) : add_result;
    assign acc_sign   = s2_eff_sub ? (s2_prod_gte_c ? s2_s_prod : s2_s_c) : s2_s_prod;
    assign acc_sticky = s2_stick_prod | s2_stick_c;

    // Pipeline register for ACC stage (also carry eff_sub for zero-sign in NRM)
    localparam ACC_DATAW = ACC_BITS + 1 + 1 + 1 + EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [ACC_DATAW-1:0] s3_data;
    VX_pipe_register #(
        .DATAW (ACC_DATAW),
        .DEPTH (ACC_LATENCY)
    ) pipe_acc (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({acc_sum,  acc_sign,  acc_sticky, s2_eff_sub, s2_max_exp, s2_frm, s2_exc}),
        .data_out(s3_data)
    );

    wire [ACC_BITS-1:0]           s3_sum;
    wire                          s3_sign;
    wire                          s3_sticky;
    wire                          s3_eff_sub;
    wire signed [EXP_IWIDTH-1:0]  s3_max_exp;
    wire [INST_FRM_BITS-1:0]      s3_frm;
    wire [3:0]                    s3_exc;

    assign {s3_sum, s3_sign, s3_sticky, s3_eff_sub, s3_max_exp, s3_frm, s3_exc} = s3_data;

    // =========================================================================
    // Stage 4: Normalize + Round (1 cycle, combinational then registered)
    // =========================================================================

    // Leading zero count on accumulated magnitude
    wire [LZC_BITS-1:0] lzc_sum;
    wire                lzc_sum_vld;
    VX_lzc #(.N(ACC_BITS)) lzc_inst (
        .data_in  (s3_sum),
        .data_out (lzc_sum),
        .valid_out(lzc_sum_vld)
    );

    wire zero_sum = ~lzc_sum_vld;

    // Normalize: shift left by lzc_sum
    // Extend to ACC_BITS+1 to catch overshift
    wire [ACC_BITS:0] sum_ext     = {1'b0, s3_sum};
    wire [ACC_BITS:0] shifted_raw = sum_ext << lzc_sum;
    wire              overshift   = shifted_raw[ACC_BITS];

    // Extract NORM_WIN_BITS-wide window: {hidden, MAN_BITS mantissa, G, R, S}
    // Normal:    bits [ACC_BITS-1 : ACC_BITS-NORM_WIN_BITS]
    // Overshift: bits [ACC_BITS   : ACC_BITS-NORM_WIN_BITS+1]
    wire [NORM_WIN_BITS-1:0] norm_window = overshift ? shifted_raw[ACC_BITS   -: NORM_WIN_BITS]
                                                     : shifted_raw[ACC_BITS-1 -: NORM_WIN_BITS];

    wire [MAN_BITS:0] norm_man   = norm_window[NORM_WIN_BITS-1:3];
    wire        guard_bit  = norm_window[2];
    wire        round_bit  = norm_window[1];
    wire        sticky_sum = norm_window[0]
                           | (overshift ? (|shifted_raw[STICK_IDX:0]) : (|shifted_raw[STICK_IDX-1:0]))
                           | s3_sticky;

    // Exponent after normalization
    // Reference: aln_prod implicit-1 at bit (ALN_BITS-2) when shift_prod=0, biased exp = max_exp.
    // After accumulation implicit-1 at bit (ACC_BITS-1 - lzc_sum).
    // After left-shift by lzc_sum, implicit-1 at bit ACC_BITS-1 of shifted_raw.
    // result_exp = max_exp + (ACC_BITS-1-lzc_sum - (ALN_BITS-2)) = max_exp - lzc_sum + 2
    localparam EXP_ADJ = 2;

    wire signed [EXP_IWIDTH-1:0] norm_exp_base  = s3_max_exp - $signed(EXP_IWIDTH'(lzc_sum)) + $signed(EXP_IWIDTH'(EXP_ADJ));
    wire signed [EXP_IWIDTH-1:0] norm_exp_ov    = norm_exp_base + $signed(EXP_IWIDTH'(1));

    wire signed [EXP_IWIDTH-1:0] pre_round_exp  = overshift ? norm_exp_ov : norm_exp_base;

    // RISC-V rounding (reuse VX_fp_rounding)
    wire [MAN_BITS:0] abs_rounded;
    wire              round_sign;
    wire              exact_zero;

    VX_fp_rounding #(.DAT_WIDTH(MAN_BITS+1)) u_round (
        .abs_value_i          (norm_man),
        .sign_i               (s3_sign),
        .round_sticky_bits_i  ({guard_bit, round_bit | sticky_sum}),
        .rnd_mode_i           (s3_frm),
        .effective_subtraction_i(s3_eff_sub),
        .abs_rounded_o        (abs_rounded),
        .sign_o               (round_sign),
        .exact_zero_o         (exact_zero)
    );

    // Detect mantissa overflow from rounding (all-1s + 1 wraps to 0).
    wire round_carry = (abs_rounded == '0) & ~zero_sum;
    wire [MAN_BITS-1:0] final_man = round_carry ? {MAN_BITS{1'b0}} : abs_rounded[MAN_BITS-1:0];
    wire signed [EXP_IWIDTH-1:0] final_exp_s = pre_round_exp + ($signed(round_carry ? EXP_IWIDTH'(1) : EXP_IWIDTH'(0)));

    // Exception detection
    wire is_nan_result  = s3_exc[3];
    wire is_inf_result  = s3_exc[2];
    wire inf_sign_result= s3_exc[1];
    wire nv_flag        = s3_exc[0];

    wire of_flag = (final_exp_s >= $signed(EXP_IWIDTH'(EXP_MAX))) & ~is_nan_result & ~is_inf_result;
    wire uf_flag = (final_exp_s <= $signed(EXP_IWIDTH'(0)))        & ~is_nan_result & ~is_inf_result & ~zero_sum & ~exact_zero;
    wire nx_flag = (guard_bit | round_bit | sticky_sum) & ~is_nan_result & ~is_inf_result;

    // Final result mux
    logic [FLOAT_BITS-1:0] nrm_result;
    always_comb begin
        if (is_nan_result) begin
            // Canonical quiet NaN: sign=0, exp=all-1s, quiet-bit=1, rest=0
            nrm_result = {1'b0, {EXP_BITS{1'b1}}, 1'b1, {(MAN_BITS-1){1'b0}}};
        end else if (is_inf_result) begin
            nrm_result = {inf_sign_result, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};
        end else if (of_flag) begin
            // Overflow → ±Inf
            nrm_result = {round_sign, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};
        end else if (zero_sum | exact_zero | uf_flag) begin
            // Zero or flush-to-zero underflow
            nrm_result = {round_sign, {(FLOAT_BITS-1){1'b0}}};
        end else begin
            nrm_result = {round_sign, final_exp_s[EXP_BITS-1:0], final_man};
        end
    end

    fflags_t nrm_fflags;
    assign nrm_fflags.NV = nv_flag;
    assign nrm_fflags.DZ = 1'b0;
    assign nrm_fflags.OF = of_flag;
    assign nrm_fflags.UF = uf_flag;
    assign nrm_fflags.NX = nx_flag | of_flag | uf_flag;

    // Pipeline register for NRM stage
    VX_pipe_register #(
        .DATAW (FLOAT_BITS + `FP_FLAGS_BITS),
        .DEPTH (NRM_LATENCY)
    ) pipe_nrm (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({nrm_result,  nrm_fflags}),
        .data_out({result,      fflags})
    );

endmodule
