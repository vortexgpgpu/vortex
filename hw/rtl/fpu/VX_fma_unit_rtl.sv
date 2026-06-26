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
// Minimum LATENCY is 7 (1 ini + 1 mul + 1 aln + 1 acc + 1 nrm + 2 rnd).

`include "VX_fpu_define.vh"

module VX_fma_unit_rtl import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 6,
    parameter MAN_BITS = 23,  // mantissa bits (excluding hidden bit): 23=F32, 52=F64
    parameter EXP_BITS = 8,   // exponent bits: 8=F32, 11=F64
    // 1: target FPGA DSP blocks for the mantissa multiply (inferred * + use_dsp
    //    hint; pipeline depth + retiming pack a DSP48 cascade).
    // 0: target ASIC standard cells (Wallace/CPA tree, area-optimal). Portable:
    //    the use_dsp attribute is ignored by ASIC synthesis tools.
    parameter USE_DSP  = 0,
    // 1: full IEEE subnormal support. 0: flush-to-zero (DAZ subnormal inputs to
    //    signed zero; results already FTZ) for area. Use 0 for relaxed paths (RTU).
    parameter SNORM_ENABLE = 1,
    // 1: detect NaN/inf + produce IEEE special results + fflags. 0: assume finite
    //    operands; drop the exception cone and tie fflags to 0 (area).
    parameter EXCEPT_ENABLE  = 1
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
    // The align barrel shifter is the FMA's routing-critical path only at the F64
    // width, so it is split into coarse+fine (2 stages) for F64 and left as a
    // single stage for F32 (which already meets timing) to avoid a needless
    // pipeline register and its area. Format is keyed off the significand width.
    localparam ALN_LATENCY = (MAN_BITS + 1 > 24) ? 2 : 1;
    localparam ACC_LATENCY = 1;
    localparam NRM_LATENCY = 1;
    // ROUND is split into 2 register stages (RND1: select-add round + IEEE
    // subnormal denormalize-shift; RND2: subnormal round-add + result pack) so the
    // deep round cloud closes 300 MHz. The extra cycle is reclaimed from the
    // over-provisioned MUL pipeline, so the externally observed LATENCY is unchanged.
    localparam RND_LATENCY = 2;
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
    wire valid_alnf = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY-2]; // last align register
    wire valid_acc = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY-1];
    wire valid_nrm = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY+ACC_LATENCY-1];
    wire valid_rnd1 = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY+ACC_LATENCY+NRM_LATENCY-1];
    wire valid_rnd  = mask_pipe[INI_LATENCY+MUL_LATENCY+ALN_LATENCY+ACC_LATENCY+NRM_LATENCY];

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
    // INIT: decode, classify, form canonical operands
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

    // DAZ: flush a subnormal operand to signed zero when subnormals are disabled.
    localparam DAZ = (SNORM_ENABLE == 0);
    function automatic [FLOAT_BITS-1:0] daz_f(input [FLOAT_BITS-1:0] v);
        daz_f = (DAZ && (v[FLOAT_BITS-2:MAN_BITS] == '0) && (|v[MAN_BITS-1:0]))
              ? {v[FLOAT_BITS-1], {(FLOAT_BITS-1){1'b0}}} : v;
    endfunction

    wire [FLOAT_BITS-1:0] op_a = daz_f(dataa);
    wire [FLOAT_BITS-1:0] op_b = is_add ? {1'b0, EXP_BITS'(EXP_BIAS), {MAN_BITS{1'b0}}} : daz_f(datab);
    wire [FLOAT_BITS-1:0] op_c = is_add ? daz_f(datab) : daz_f(datac);

    wire        s_a0 = op_a[FLOAT_BITS-1] ^ s_prod_neg;
    wire        s_b0 = op_b[FLOAT_BITS-1];
    // MUL: the artificial zero addend takes the PRODUCT's sign (s_a^s_b) so a zero
    // product keeps the IEEE multiply sign (e.g. (+0)*(-0) = -0); adding a same-
    // signed zero is a no-op for nonzero products and avoids the add zero-sign rule.
    wire        s_c0 = is_mul ? (op_a[FLOAT_BITS-1] ^ op_b[FLOAT_BITS-1])
                              : (op_c[FLOAT_BITS-1] ^ s_c_neg);

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
    // MUL's artificial zero addend must never win the prod/c alignment, else a
    // product with a negative biased exponent (a representable subnormal) would be
    // shifted out and flushed to zero. Use a sentinel exponent below any product
    // exponent (-EXP_BIAS <= min exp_prod0); the +-0 addend then aligns away
    // harmlessly and the product drives max_exp.
    wire signed [EXP_IWIDTH-1:0] exp_c = is_mul ? -$signed(EXP_IWIDTH'(EXP_BIAS))
                                        : clss_c0.is_subnormal ? EXP_IWIDTH'(1)
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
    // EXCEPT_ENABLE=0 assumes finite operands: drop the NaN/inf cone (and fflags).
    wire [3:0] exc0 = EXCEPT_ENABLE ? {result_nan, result_inf, result_inf_sign, early_nv} : 4'b0;

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
    // MUL: MUL_LATENCY cycles — significand multiply (product path) with the
    // operand/exponent side-band piped in parallel.
    // =========================================================================

    wire [PROD_BITS-1:0] s1_prod;

    // Wide multiplies (F64 53x53) on FPGA: a flat a*b maps to a DSP48 cascade
    // whose partial sums chain combinationally over PCOUT->PCIN, which cannot
    // meet 300MHz. Splitting operand B and REGISTERING each partial product
    // forces a registered DSP output per segment, breaking the cascade into
    // short pipelined hops. Portable: pure RTL + a 'use_dsp' hint (ASIC ignores
    // it and uses the Wallace/inferred path below).
    localparam SPLIT_MUL = (USE_DSP != 0) && (SIG_BITS > 24) && (MUL_LATENCY >= 2);

    if (SPLIT_MUL) begin : g_mul_dsp_split
        localparam BL_W = SIG_BITS - (SIG_BITS/2);   // low chunk of B
        localparam BH_W = SIG_BITS/2;                // high chunk of B
        (* use_dsp = "yes" *) wire [SIG_BITS+BL_W-1:0] pp_lo = r1_sig_a * r1_sig_b[BL_W-1:0];
        (* use_dsp = "yes" *) wire [SIG_BITS+BH_W-1:0] pp_hi = r1_sig_a * r1_sig_b[SIG_BITS-1:BL_W];
        reg [SIG_BITS+BL_W-1:0] pp_lo_q;
        reg [SIG_BITS+BH_W-1:0] pp_hi_q;
        always @(posedge clk) if (enable) begin pp_lo_q <= pp_lo; pp_hi_q <= pp_hi; end // DSP MREG/PREG
        reg [PROD_BITS-1:0] prod_q;
        always @(posedge clk) if (enable) prod_q <= PROD_BITS'(pp_lo_q) + (PROD_BITS'(pp_hi_q) << BL_W);
        // 2 stages consumed above; pad the remainder of MUL_LATENCY.
        VX_pipe_register #(.DATAW(PROD_BITS), .DEPTH(MUL_LATENCY-2)) pm (
            .clk(clk), .reset(reset), .enable(enable), .data_in(prod_q), .data_out(s1_prod));
    end else if (USE_DSP) begin : g_mul_dsp
        (* use_dsp = "yes" *) wire [PROD_BITS-1:0] dsp_prod = PROD_BITS'(r1_sig_a) * PROD_BITS'(r1_sig_b);
        VX_pipe_register #(.DATAW(PROD_BITS), .DEPTH(MUL_LATENCY)) pm (
            .clk(clk), .reset(reset), .enable(enable), .data_in(dsp_prod), .data_out(s1_prod));
    end else if (MUL_LATENCY < `LATENCY_IMUL && SIG_BITS <= 24) begin : g_mul_wallace
        wire [PROD_BITS-1:0] wal_prod;
        VX_wallace_mul #(
            .N (SIG_BITS), .P (PROD_BITS), .CPA_KS(!`FORCE_BUILTIN_ADDER(PROD_BITS))
        ) u_mul (.a(r1_sig_a), .b(r1_sig_b), .p(wal_prod));
        VX_pipe_register #(.DATAW(PROD_BITS), .DEPTH(MUL_LATENCY)) pm (
            .clk(clk), .reset(reset), .enable(enable), .data_in(wal_prod), .data_out(s1_prod));
    end else begin : g_mul_infer
        wire [PROD_BITS-1:0] inf_prod = PROD_BITS'(r1_sig_a) * PROD_BITS'(r1_sig_b);
        VX_pipe_register #(.DATAW(PROD_BITS), .DEPTH(MUL_LATENCY)) pm (
            .clk(clk), .reset(reset), .enable(enable), .data_in(inf_prod), .data_out(s1_prod));
    end

    // Side-band (exponents, addend, flags) delayed to match the product latency.
    localparam SIDE_W = EXP_IWIDTH + 1 + SIG_BITS + EXP_IWIDTH + 1 + INST_FRM_BITS + 4;
    wire signed [EXP_IWIDTH-1:0]  s1_exp_prod;
    wire                          s1_s_prod;
    wire [SIG_BITS-1:0]           s1_sig_c;
    wire signed [EXP_IWIDTH-1:0]  s1_exp_c;
    wire                          s1_s_c;
    wire [INST_FRM_BITS-1:0]      s1_frm;
    wire [3:0]                    s1_exc;
    wire [SIDE_W-1:0] s1_side;
    VX_pipe_register #(
        .DATAW (SIDE_W),
        .DEPTH (MUL_LATENCY)
    ) pipe_side (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({r1_exp_prod, r1_s_prod, r1_sig_c, r1_exp_c, r1_s_c, r1_frm, r1_exc}),
        .data_out(s1_side)
    );
    assign {s1_exp_prod, s1_s_prod, s1_sig_c, s1_exp_c, s1_s_c, s1_frm, s1_exc} = s1_side;

    // =========================================================================
    // ALIGN: align product and C addend
    //   Single barrel shifter — only the smaller operand is shifted.
    //   Magnitude compare deferred to ACC stage.
    // =========================================================================

    wire s1_eff_sub = s1_s_prod ^ s1_s_c;

    // Exponent comparison and shift amount
    wire prod_ge_c = (s1_exp_prod >= s1_exp_c);
    wire signed [EXP_IWIDTH-1:0] s1_max_exp = prod_ge_c ? s1_exp_prod : s1_exp_c;

    localparam SHIFT_BITS = `LOG2UP(ALN_BITS + 1);
    localparam FINE_BITS  = 4;   // align barrel shifter is split coarse/fine on this boundary
    wire signed [EXP_IWIDTH-1:0] exp_diff = prod_ge_c ? (s1_exp_prod - s1_exp_c)
                                                       : (s1_exp_c - s1_exp_prod);
    wire [SHIFT_BITS-1:0] shift_amt = (exp_diff > $signed(EXP_IWIDTH'(ALN_BITS)))
                                    ? SHIFT_BITS'(ALN_BITS)
                                    : SHIFT_BITS'(exp_diff);

    // Extend product into ALN_BITS-wide field with 3 GRS guard bits at bottom
    wire [ALN_BITS-1:0] prod_aligned_full = {s1_prod, {(ALN_BITS-PROD_BITS){1'b0}}};

    // Place C significand aligned with product: implicit-1 at bit (ALN_BITS-2)
    wire [ALN_BITS-1:0] c_aligned_full = {1'b0, s1_sig_c, {(ALN_BITS-SIG_BITS-1){1'b0}}};

    // Shift the smaller operand right; the larger one passes through unchanged.
    // The full variable shift is split into a coarse stage (high shift bits) and
    // a fine stage (low FINE_BITS), with a register between them: a single wide
    // barrel shifter is the FMA's routing-critical path at the F64 width, so two
    // smaller shifters close timing. Exact: (x >> coarse) >> fine == x >> shift_amt;
    // sticky is taken from the fully-shifted value, so the split is bit-accurate.
    wire [ALN_BITS-1:0] shift_in = prod_ge_c ? c_aligned_full : prod_aligned_full;
    wire [ALN_BITS-1:0] fixed_op = prod_ge_c ? prod_aligned_full : c_aligned_full;

    // Common align outputs (both implementations drive these; consumed by the
    // ALIGN → ACC register below).
    wire [ALN_BITS-1:0]          aln_prod, aln_c;
    wire                         aln_sticky;
    wire                         aln_eff_sub, aln_s_prod, aln_s_c;
    wire signed [EXP_IWIDTH-1:0] aln_max_exp;
    wire [INST_FRM_BITS-1:0]     aln_frm;
    wire [3:0]                   aln_exc;

    if (ALN_LATENCY == 2) begin : g_align_split
        // F64: split the variable shift into a coarse stage (high shift bits)
        // and a fine stage (low FINE_BITS) with a register between them — two
        // smaller shifters close timing. Exact: (x >> coarse) >> fine == x >> amt;
        // sticky is taken from the fully-shifted value, so the split is bit-accurate.
        wire valid_aln = mask_pipe[INI_LATENCY+MUL_LATENCY-1]; // coarse-shift register
        wire [SHIFT_BITS-1:0] coarse_amt = {shift_amt[SHIFT_BITS-1:FINE_BITS], {FINE_BITS{1'b0}}};
        wire [2*ALN_BITS-1:0] shift_ext  = {shift_in, {ALN_BITS{1'b0}}};
        wire [2*ALN_BITS-1:0] coarse_sh  = shift_ext >> coarse_amt;

        // ALIGN stage A → B register (coarse result + remaining fine amount + ctrl)
        localparam ALN1_DATAW = 2*ALN_BITS + FINE_BITS + ALN_BITS + 1 + 1 + 1 + 1 + EXP_IWIDTH + INST_FRM_BITS + 4;
        wire [ALN1_DATAW-1:0] aln1_data;
        VX_pipe_register #(
            .DATAW (ALN1_DATAW),
            .DEPTH (1)
        ) pipe_aln1 (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable && valid_aln),
            .data_in ({coarse_sh, shift_amt[FINE_BITS-1:0], fixed_op, prod_ge_c, s1_eff_sub, s1_s_prod, s1_s_c, s1_max_exp, s1_frm, s1_exc}),
            .data_out(aln1_data)
        );

        wire [2*ALN_BITS-1:0]        a1_coarse;
        wire [FINE_BITS-1:0]         a1_fine;
        wire [ALN_BITS-1:0]          a1_fixed;
        wire                         a1_prod_ge_c, a1_eff_sub, a1_s_prod, a1_s_c;
        wire signed [EXP_IWIDTH-1:0] a1_max_exp;
        wire [INST_FRM_BITS-1:0]     a1_frm;
        wire [3:0]                   a1_exc;
        assign {a1_coarse, a1_fine, a1_fixed, a1_prod_ge_c, a1_eff_sub, a1_s_prod, a1_s_c, a1_max_exp, a1_frm, a1_exc} = aln1_data;

        // Fine shift completes the alignment; sticky from the residual low bits.
        wire [2*ALN_BITS-1:0] fine_sh = a1_coarse >> a1_fine;
        wire [ALN_BITS-1:0] shift_out = fine_sh[2*ALN_BITS-1 : ALN_BITS];

        assign aln_sticky  = |fine_sh[ALN_BITS-1:0];
        assign aln_prod    = a1_prod_ge_c ? a1_fixed  : shift_out;
        assign aln_c       = a1_prod_ge_c ? shift_out : a1_fixed;
        assign aln_eff_sub = a1_eff_sub;
        assign aln_s_prod  = a1_s_prod;
        assign aln_s_c     = a1_s_c;
        assign aln_max_exp = a1_max_exp;
        assign aln_frm     = a1_frm;
        assign aln_exc     = a1_exc;
    end else begin : g_align_single
        // F32: single barrel shifter — not routing-critical, no inter-shift register.
        wire [2*ALN_BITS-1:0] shift_ext = {shift_in, {ALN_BITS{1'b0}}};
        wire [2*ALN_BITS-1:0] full_sh   = shift_ext >> shift_amt;
        wire [ALN_BITS-1:0] shift_out   = full_sh[2*ALN_BITS-1 : ALN_BITS];

        assign aln_sticky  = |full_sh[ALN_BITS-1:0];
        assign aln_prod    = prod_ge_c ? fixed_op : shift_out;
        assign aln_c       = prod_ge_c ? shift_out : fixed_op;
        assign aln_eff_sub = s1_eff_sub;
        assign aln_s_prod  = s1_s_prod;
        assign aln_s_c     = s1_s_c;
        assign aln_max_exp = s1_max_exp;
        assign aln_frm     = s1_frm;
        assign aln_exc     = s1_exc;
    end

    // =========================================================================
    // ALIGN → ACC register
    // =========================================================================
    localparam ALN_DATAW = ALN_BITS + ALN_BITS + 1 + 1 + 1 + 1 + EXP_IWIDTH + INST_FRM_BITS + 4;

    wire [ALN_DATAW-1:0] s2_data;
    VX_pipe_register #(
        .DATAW (ALN_DATAW),
        .DEPTH (1)
    ) pipe_aln (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_alnf),
        .data_in ({aln_prod, aln_c, aln_sticky, aln_eff_sub, aln_s_prod, aln_s_c, aln_max_exp, aln_frm, aln_exc}),
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
    // ACC: accumulate + LZC
    //   Magnitude compare performed here.
    //   Exact LZC on acc_sum provides shift count for NORM stage.
    // =========================================================================

    // Magnitude comparison
    wire prod_gte_c = (s2_aln_prod >= s2_aln_c);

    // Addition and subtraction paths
    wire [ACC_BITS-1:0] add_result = {1'b0, s2_aln_prod} + {1'b0, s2_aln_c};
    wire [ACC_BITS-1:0] sub_ab     = {1'b0, s2_aln_prod} - {1'b0, s2_aln_c};
    wire [ACC_BITS-1:0] sub_mag    = prod_gte_c ? sub_ab : (~sub_ab + ACC_BITS'(1));

    // Effective subtraction: bits shifted out of the smaller operand (s2_sticky)
    // lie below the kept window and must BORROW from the magnitude; the residual
    // (1 ULP - tail) keeps sticky set. Without this borrow the rounder sees the
    // windowed magnitude (too large by up to ~1 ULP), so directed rounding can
    // wrongly round the result up — even to infinity (e.g. 1.0 + (-3.4e38), RDN).
    wire sub_borrow = s2_eff_sub & s2_sticky;

    // For subtraction with |C| > |prod|: negate (serial, OK — LZA is slower)
    wire [ACC_BITS-1:0] acc_sum;
    wire                acc_sign;

    assign acc_sum  = s2_eff_sub ? (sub_mag - ACC_BITS'(sub_borrow)) : add_result;
    assign acc_sign = s2_eff_sub ? (prod_gte_c ? s2_s_prod : s2_s_c) : s2_s_prod;

    // Leading zero count on accumulated result; provides shift count for normalization.
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
    // NORM: normalize using registered LZC prediction
    //   Barrel shift only — LZC hoisted into ACC stage.
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
    // ROUND1: select-add rounding (normal path) + IEEE subnormal denormalize-shift.
    //   man+1 computed in parallel with the round decision; the normal exponent and
    //   overflow result are resolved here. The subnormal datapath denormalizes
    //   rnd_man (right-shift by 1-exp_norm) and resolves its guard/sticky here too;
    //   the round-add itself is deferred to ROUND2. The two halves balance the deep
    //   round cloud across a register so each closes 300 MHz.
    //   SNORM_ENABLE=0 leaves gen_sub=0 so the subnormal datapath prunes -> FTZ.
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

    // --- Subnormal denormalize-shift (round-add deferred to ROUND2) ---
    wire signed [EXP_IWIDTH-1:0] exp_norm = r5_overshift ? r5_exp_plus1 : r5_exp_base;
    wire result_sub = ($signed(exp_norm) <= 0) & ~is_nan_result & ~is_inf_result & ~r5_zero_sum & ~exact_zero;
    wire gen_sub    = (SNORM_ENABLE != 0) & result_sub;

    localparam SH_W = `CLOG2(SIG_BITS + 2) + 1;
    wire signed [EXP_IWIDTH-1:0] sub_amt = gen_sub ? (EXP_IWIDTH'(1) - exp_norm) : '0;
    wire huge_sub = gen_sub & ($signed(sub_amt) >= $signed(EXP_IWIDTH'(SIG_BITS + 1)));
    wire [SH_W-1:0] sdsh = huge_sub ? SH_W'(SIG_BITS) : SH_W'(sub_amt);

    wire [SIG_BITS-1:0] sbelow = (sdsh <= 1) ? '0 : ((SIG_BITS'(1) << (sdsh - 1)) - SIG_BITS'(1));
    wire [SIG_BITS-1:0] sub_man = rnd_man >> sdsh;
    wire sub_g = (sdsh == 0) ? guard_bit : huge_sub ? 1'b0 : rnd_man[sdsh - 1];
    wire sub_s = (sdsh == 0) ? (round_bit | sticky_sum)
               : ((huge_sub ? (|rnd_man) : (|(rnd_man & sbelow))) | guard_bit | round_bit | sticky_sum);

    wire sub_inexact = sub_g | sub_s;
    wire ftz_flush = result_sub & (SNORM_ENABLE == 0);          // FTZ when subnormals disabled

    wire uf_flag = result_sub & sub_inexact & ~is_nan_result & ~is_inf_result;
    wire nx_flag = (gen_sub ? sub_inexact : (guard_bit | round_bit | sticky_sum)) & ~is_nan_result & ~is_inf_result;

    // On overflow, IEEE/RISC-V picks the magnitude-largest finite vs infinity per
    // rounding mode and sign: RTZ always -> max-normal; RDN -> max-normal for
    // positive (else -inf); RUP -> max-normal for negative (else +inf); RNE/RMM
    // -> infinity. (The old code always returned infinity, breaking directed RTL.)
    wire ovf_to_max = (r5_frm == INST_FRM_RTZ)
                    | (r5_frm == INST_FRM_RDN & ~round_sign)
                    | (r5_frm == INST_FRM_RUP &  round_sign);
    wire [FLOAT_BITS-1:0] ovf_result = ovf_to_max
        ? {round_sign, {(EXP_BITS-1){1'b1}}, 1'b0, {MAN_BITS{1'b1}}} // largest finite
        : {round_sign, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};          // infinity

    // =========================================================================
    // ROUND1 → ROUND2 register
    // =========================================================================
    localparam RND1_DATAW = MAN_BITS + EXP_BITS + 1 + FLOAT_BITS + 1 + 1
                          + SIG_BITS + 2 + INST_FRM_BITS + 1 + 3 + 3 + 3;

    wire [RND1_DATAW-1:0] s5_data;
    VX_pipe_register #(
        .DATAW (RND1_DATAW),
        .DEPTH (1)
    ) pipe_rnd1 (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_rnd1),
        .data_in ({final_man, final_exp_s[EXP_BITS-1:0], of_flag, ovf_result, round_sign, gen_sub,
                   sub_man, sub_g, sub_s, r5_frm, r5_sign,
                   is_nan_result, is_inf_result, inf_sign_result,
                   r5_zero_sum, exact_zero, ftz_flush,
                   nv_flag, uf_flag, nx_flag}),
        .data_out(s5_data)
    );

    wire [MAN_BITS-1:0]      q_final_man;
    wire [EXP_BITS-1:0]      q_final_exp;
    wire                     q_of_flag;
    wire [FLOAT_BITS-1:0]    q_ovf_result;
    wire                     q_round_sign;
    wire                     q_gen_sub;
    wire [SIG_BITS-1:0]      q_sub_man;
    wire                     q_sub_g, q_sub_s;
    wire [INST_FRM_BITS-1:0] q_frm;
    wire                     q_sign;
    wire                     q_is_nan, q_is_inf, q_inf_sign;
    wire                     q_zero_sum, q_exact_zero, q_ftz_flush;
    wire                     q_nv, q_uf, q_nx;
    assign {q_final_man, q_final_exp, q_of_flag, q_ovf_result, q_round_sign, q_gen_sub,
            q_sub_man, q_sub_g, q_sub_s, q_frm, q_sign,
            q_is_nan, q_is_inf, q_inf_sign,
            q_zero_sum, q_exact_zero, q_ftz_flush,
            q_nv, q_uf, q_nx} = s5_data;

    // =========================================================================
    // ROUND2: subnormal round-add + final result mux/pack.
    // =========================================================================

    // Round the denormalized mantissa (same rule as the normal select-add round).
    reg sub_round_up;
    wire [1:0] q_sub_gs = {q_sub_g, q_sub_s};
    always @(*) begin
        case (q_frm)
            INST_FRM_RNE: sub_round_up = (q_sub_gs == 2'b11) | ((q_sub_gs == 2'b10) & q_sub_man[0]);
            INST_FRM_RTZ: sub_round_up = 1'b0;
            INST_FRM_RDN: sub_round_up = (|q_sub_gs) &  q_sign;
            INST_FRM_RUP: sub_round_up = (|q_sub_gs) & ~q_sign;
            INST_FRM_RMM: sub_round_up = q_sub_gs[1];
            default:      sub_round_up = 1'bx;
        endcase
    end
    wire [MAN_BITS:0] sub_abs = q_sub_man + (sub_round_up ? (MAN_BITS+1)'(1) : '0);
    wire              sub_to_normal = sub_abs[MAN_BITS];        // rounded up to smallest normal
    wire [MAN_BITS-1:0] sub_man_final = sub_abs[MAN_BITS-1:0];
    wire [EXP_BITS-1:0] sub_exp_final = sub_to_normal ? EXP_BITS'(1) : '0;

    // Final result mux
    logic [FLOAT_BITS-1:0] rnd_result;
    always_comb begin
        if (q_is_nan) begin
            rnd_result = {1'b0, {EXP_BITS{1'b1}}, 1'b1, {(MAN_BITS-1){1'b0}}};
        end else if (q_is_inf) begin
            rnd_result = {q_inf_sign, {EXP_BITS{1'b1}}, {MAN_BITS{1'b0}}};
        end else if (q_of_flag) begin
            rnd_result = q_ovf_result;
        end else if (q_gen_sub) begin
            rnd_result = {q_round_sign, sub_exp_final, sub_man_final}; // emitted subnormal
        end else if (q_zero_sum | q_exact_zero | q_ftz_flush) begin
            rnd_result = {q_round_sign, {(FLOAT_BITS-1){1'b0}}};
        end else begin
            rnd_result = {q_round_sign, q_final_exp, q_final_man};
        end
    end

    fflags_t rnd_fflags;
    assign rnd_fflags.NV = EXCEPT_ENABLE & q_nv;
    assign rnd_fflags.DZ = 1'b0;
    assign rnd_fflags.OF = EXCEPT_ENABLE & q_of_flag;
    assign rnd_fflags.UF = EXCEPT_ENABLE & q_uf;
    assign rnd_fflags.NX = EXCEPT_ENABLE & (q_nx | q_of_flag | q_uf);

    // =========================================================================
    // ROUND2 output register
    // =========================================================================
    VX_pipe_register #(
        .DATAW (FLOAT_BITS + `FP_FLAGS_BITS),
        .DEPTH (1)
    ) pipe_rnd (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable && valid_rnd),
        .data_in ({rnd_result, rnd_fflags}),
        .data_out({result,     fflags})
    );

endmodule
