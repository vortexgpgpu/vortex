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

// Single-lane FP32 division pipeline: Non-Restoring Radix-2 × 2 steps/stage.
// Pipeline: 1 PRE + 13 SRT + 1 CONV + 1 NRM = 16 cycles = LATENCY_FDIV.
// Supports all RISC-V rounding modes (RNE/RTZ/RDN/RUP/RMM).
// Produces fflags: NV, DZ, OF, UF, NX.  F32-only; fmt is ignored.
//
// Algorithm (non-restoring, 2 bits/stage):
//   W[0] = (sig_a - q_int * sig_b) * 32   q_int ∈ {0,1}
//   Each SRT stage:
//     Step A: q_a = (W≥0); W = 2W + (q_a ? -D : +D)
//     Step B: q_b = (W≥0); W = 2W + (q_b ? -D : +D)
//   After 13 stages (26 NR bits):
//     Q_frac = 2 * q_bits_uint - (2^26 - 1)
//     Q_total[26] = q_int; Q_total[25:0] = Q_frac[25:0]
//     if W_final < 0: Q_corrected = Q_total - 1; W_corr = W_final + D
//     else:           Q_corrected = Q_total;      W_corr = W_final
//     sticky = (W_corr != 0)
//   Normalize: if q_int=1 use Q_corrected as-is; else shift left 1 (Q≥0.5 always)
//   Round: VX_fp_rounding on 24-bit mantissa with G/R/S bits.

`include "VX_fpu_define.vh"

`ifdef FPU_TYPE_DSP

module VX_fdiv_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,

    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [31:0] dataa,   // dividend
    input  wire [31:0] datab,   // divisor

    output wire [31:0]               result,
    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    `UNUSED_VAR (fmt)   // F32-only

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam MAN_BITS    = 23;
    localparam EXP_BITS    = 8;
    localparam SIG_BITS    = MAN_BITS + 1;          // 24
    localparam SUBLZC_BITS = `LOG2UP(SIG_BITS);     // 5
    localparam W_BITS      = 30;                    // signed partial-remainder width
    localparam NR_BITS     = 26;                    // total NR iterations (2 per stage)
    localparam NR_STAGES   = NR_BITS / 2;           // 13

    // =========================================================================
    // Stage-state packing layout (106-bit flat vector, MSB→LSB):
    //   W[29:0]  | D[29:0]  | q_bits[25:0] | q_int | exp_r[9:0] | sign_r | frm[2:0] | exc[4:0]
    //   30         30          26              1       10            1        3           5  = 106
    // =========================================================================
    localparam EXC_LO  = 0;
    localparam FRM_LO  = EXC_LO  + 5;
    localparam SGN_LO  = FRM_LO  + INST_FRM_BITS;  // 8
    localparam EXP_LO  = SGN_LO  + 1;              // 9
    localparam QI_LO   = EXP_LO  + 10;             // 19
    localparam QB_LO   = QI_LO   + 1;              // 20
    localparam D_LO    = QB_LO   + NR_BITS;        // 46
    localparam W_LO    = D_LO    + W_BITS;         // 76
    localparam STAGE_W = W_LO    + W_BITS;         // 106

    // =========================================================================
    // Stage 0 (PRE, combinational): classify, normalize subnormals, compute W[0]
    // =========================================================================
    wire        s_a = dataa[31];
    wire [7:0]  e_a = dataa[30:23];
    wire [22:0] m_a = dataa[22:0];
    wire        s_b = datab[31];
    wire [7:0]  e_b = datab[30:23];
    wire [22:0] m_b = datab[22:0];

    fclass_t clss_a, clss_b;
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_a (
        .exp_i(e_a), .man_i(m_a), .clss_o(clss_a));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_b (
        .exp_i(e_b), .man_i(m_b), .clss_o(clss_b));
    `UNUSED_VAR ({clss_a.is_normal, clss_a.is_quiet,
                  clss_b.is_normal, clss_b.is_quiet})

    // Significands with implicit leading bit
    wire [SIG_BITS-1:0] sig_a_raw = {~(clss_a.is_zero | clss_a.is_subnormal), m_a};
    wire [SIG_BITS-1:0] sig_b_raw = {~(clss_b.is_zero | clss_b.is_subnormal), m_b};

    // LZC for subnormal normalization
    wire [SUBLZC_BITS-1:0] lzc_a, lzc_b;
    wire lzc_a_vld, lzc_b_vld;
    VX_lzc #(.N(SIG_BITS)) lzc_suba (.data_in(sig_a_raw), .data_out(lzc_a), .valid_out(lzc_a_vld));
    VX_lzc #(.N(SIG_BITS)) lzc_subb (.data_in(sig_b_raw), .data_out(lzc_b), .valid_out(lzc_b_vld));
    `UNUSED_VAR ({lzc_a_vld, lzc_b_vld})

    wire [SIG_BITS-1:0] sig_a_norm = clss_a.is_subnormal ? (sig_a_raw << lzc_a) : sig_a_raw;
    wire [SIG_BITS-1:0] sig_b_norm = clss_b.is_subnormal ? (sig_b_raw << lzc_b) : sig_b_raw;

    // Adjusted biased exponents (10-bit signed, matching VX_fma_unit convention)
    wire signed [9:0] exp_a = clss_a.is_subnormal ? (10'sd1 - 10'(lzc_a))
                            : clss_a.is_zero       ? 10'sd0
                            :                        10'(e_a);
    wire signed [9:0] exp_b = clss_b.is_subnormal ? (10'sd1 - 10'(lzc_b))
                            : clss_b.is_zero       ? 10'sd0
                            :                        10'(e_b);

    // Result sign and raw biased exponent (q_int adjustment applied in CONV)
    wire        sign_r0 = s_a ^ s_b;
    wire signed [9:0] exp_r0 = exp_a - exp_b + 10'sd127;

    // Special-case detection
    wire nan_a  = clss_a.is_nan,  nan_b  = clss_b.is_nan;
    wire inf_a  = clss_a.is_inf,  inf_b  = clss_b.is_inf;
    wire zero_a = clss_a.is_zero, zero_b = clss_b.is_zero;
    wire snan_a = clss_a.is_signaling, snan_b = clss_b.is_signaling;

    wire nv0 = snan_a | snan_b | (zero_a & zero_b) | (inf_a & inf_b); // Invalid
    wire dz0 = zero_b & ~nan_a & ~nan_b & ~zero_a;                    // Divide-by-zero

    // exc[4:0]: [4]=result_nan [3]=result_inf [2]=result_zero [1]=dz [0]=nv
    wire result_nan0  = nan_a | nan_b | nv0;
    wire result_inf0  = (inf_a | zero_b) & ~result_nan0;
    wire result_zero0 = (zero_a | inf_b) & ~result_nan0;
    wire [4:0] exc0 = {result_nan0, result_inf0, result_zero0, dz0, nv0};

    // Divisor D = sig_b_norm * 32  (30-bit unsigned, D ∈ [2^28, 2^29))
    wire [W_BITS-1:0] D0 = {1'b0, sig_b_norm, 5'b0};   // W_BITS=30: 1+24+5

    // Integer quotient bit: 1 iff sig_a ≥ sig_b  (Q ∈ [1,2)); else Q ∈ (0.5,1)
    wire q_int0 = (sig_a_norm >= sig_b_norm);

    // Initial remainder W[0] = (sig_a - q_int*sig_b) * 32, always in [0, D)
    // Result fits in SIG_BITS (no overflow: sig_a < 2*sig_b for normalized significands)
    wire [SIG_BITS-1:0] pre_diff =
        sig_a_norm - (q_int0 ? sig_b_norm : {SIG_BITS{1'b0}});
    wire [W_BITS-1:0] W0 = {1'b0, pre_diff, 5'b0};  // positive, bit 29=0

    // =========================================================================
    // PRE pipeline register (latency cycle 1)
    // =========================================================================
    wire [STAGE_W-1:0] srt_stage [0:NR_STAGES];

    wire [STAGE_W-1:0] pre_in;
    assign pre_in[W_LO  +: W_BITS]      = W0;
    assign pre_in[D_LO  +: W_BITS]      = D0;
    assign pre_in[QB_LO +: NR_BITS]     = {NR_BITS{1'b0}};
    assign pre_in[QI_LO]                = q_int0;
    assign pre_in[EXP_LO +: 10]         = exp_r0;
    assign pre_in[SGN_LO]               = sign_r0;
    assign pre_in[FRM_LO +: INST_FRM_BITS] = frm;
    assign pre_in[EXC_LO +: 5]          = exc0;

    VX_pipe_register #(
        .DATAW (STAGE_W),
        .DEPTH (1)
    ) pre_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in (pre_in),
        .data_out(srt_stage[0])
    );

    // =========================================================================
    // SRT stages 1..13 (latency cycles 2..14)
    // Each stage performs two non-restoring radix-2 steps.
    // NR invariant: |W| ≤ D throughout.
    // =========================================================================
    for (genvar k = 0; k < NR_STAGES; k++) begin : g_srt

        // Unpack input state
        wire [W_BITS-1:0]        W_in     = srt_stage[k][W_LO  +: W_BITS];
        wire [W_BITS-1:0]        D_in     = srt_stage[k][D_LO  +: W_BITS];
        wire [NR_BITS-1:0]       qb_in    = srt_stage[k][QB_LO +: NR_BITS];
        wire                     qi_in    = srt_stage[k][QI_LO];
        wire [9:0]               exp_in   = srt_stage[k][EXP_LO +: 10];
        wire                     sgn_in   = srt_stage[k][SGN_LO];
        wire [INST_FRM_BITS-1:0] frm_in   = srt_stage[k][FRM_LO +: INST_FRM_BITS];
        wire [4:0]               exc_in   = srt_stage[k][EXC_LO +: 5];

        // D sign-extended to W_BITS+1 for intermediate arithmetic
        wire [W_BITS:0] D31 = {1'b0, D_in};  // D is unsigned, D[29]=0

        // NR Step A: q_a = (W_in ≥ 0); W_a = 2*W_in + (q_a ? -D : +D)
        // 31-bit intermediate; NR invariant ensures result fits in 30-bit signed (lower bits)
        wire q_a = ~W_in[W_BITS-1];                          // 1 if W_in ≥ 0
        wire [W_BITS:0] Wa31 = q_a ? ({W_in, 1'b0} - D31)   // 2*W_in - D
                                   : ({W_in, 1'b0} + D31);   // 2*W_in + D
        `UNUSED_VAR (Wa31[W_BITS])                            // carry: invariant guarantees unused
        wire [W_BITS-1:0] W_a = Wa31[W_BITS-1:0];

        // NR Step B: q_b = (W_a ≥ 0); W_b = 2*W_a + (q_b ? -D : +D)
        wire q_b = ~W_a[W_BITS-1];
        wire [W_BITS:0] Wb31 = q_b ? ({W_a, 1'b0} - D31)
                                   : ({W_a, 1'b0} + D31);
        `UNUSED_VAR (Wb31[W_BITS])
        wire [W_BITS-1:0] W_b = Wb31[W_BITS-1:0];

        // Shift q_bits left by 2, fill from LSB: q_a at bit 1, q_b at bit 0
        // Top 2 bits of qb_in are shifted out (oldest NR bits, no longer needed)
        `UNUSED_VAR (qb_in[NR_BITS-1:NR_BITS-2])
        wire [NR_BITS-1:0] qb_new = {qb_in[NR_BITS-3:0], q_a, q_b};

        // Pack output state
        wire [STAGE_W-1:0] s_out;
        assign s_out[W_LO  +: W_BITS]         = W_b;
        assign s_out[D_LO  +: W_BITS]         = D_in;
        assign s_out[QB_LO +: NR_BITS]        = qb_new;
        assign s_out[QI_LO]                   = qi_in;
        assign s_out[EXP_LO +: 10]            = exp_in;
        assign s_out[SGN_LO]                  = sgn_in;
        assign s_out[FRM_LO +: INST_FRM_BITS] = frm_in;
        assign s_out[EXC_LO +: 5]             = exc_in;

        VX_pipe_register #(
            .DATAW (STAGE_W),
            .DEPTH (1)
        ) srt_reg (
            .clk(clk), .reset(reset), .enable(enable),
            .data_in (s_out),
            .data_out(srt_stage[k+1])
        );
    end

    // =========================================================================
    // CONV (combinational after srt_stage[NR_STAGES]): NR→binary + round/sticky
    // =========================================================================
    wire [W_BITS-1:0]        W_cv   = srt_stage[NR_STAGES][W_LO  +: W_BITS];
    wire [W_BITS-1:0]        D_cv   = srt_stage[NR_STAGES][D_LO  +: W_BITS];
    wire [NR_BITS-1:0]       qb_cv  = srt_stage[NR_STAGES][QB_LO +: NR_BITS];
    wire                     qi_cv  = srt_stage[NR_STAGES][QI_LO];
    wire [9:0]               exp_cv = srt_stage[NR_STAGES][EXP_LO +: 10];
    wire                     sgn_cv = srt_stage[NR_STAGES][SGN_LO];
    wire [INST_FRM_BITS-1:0] frm_cv = srt_stage[NR_STAGES][FRM_LO +: INST_FRM_BITS];
    wire [4:0]               exc_cv = srt_stage[NR_STAGES][EXC_LO +: 5];

    // NR → binary:  Q_frac = 2*q_bits_uint - (2^NR_BITS - 1)
    // Range: [1, 2^NR_BITS-1] — always positive because first NR bit is always 1
    // (W[0] ≥ 0 always → q_a of stage 1 = 1 → qb_cv[NR_BITS-1] = 1 → q_bits_uint ≥ 2^(NR_BITS-1))
    wire [NR_BITS:0] Q_frac = {qb_cv, 1'b0} - {1'b0, {NR_BITS{1'b1}}};  // 2*P - (2^26-1)
    // Q_frac[NR_BITS] = 0 always (max = 2^NR_BITS-1 < 2^NR_BITS)
    `UNUSED_VAR (Q_frac[NR_BITS])

    // Full 27-bit quotient in Q1.26 (q_int=1) or Q0.26 (q_int=0) format
    wire [NR_BITS:0] Q_tot = {qi_cv, Q_frac[NR_BITS-1:0]};

    // NR correction: if W_final < 0, quotient was over-estimated by one ULP
    wire              W_neg  = W_cv[W_BITS-1];
    wire [NR_BITS:0]  Q_corr = W_neg ? (Q_tot - 1'b1) : Q_tot;

    // Corrected remainder for sticky bit (Q_corr ≥ 0 after correction)
    wire [W_BITS-1:0] W_corr  = W_neg ? (W_cv + D_cv) : W_cv;
    wire              sticky_r = (W_corr != '0);

    // Normalize: place implicit-1 at bit NR_BITS (= 26).
    //   q_int=1: Q_corr[26]=1 already
    //   q_int=0: Q_corr[25]=1 (Q ≥ 0.5 since sig_a/sig_b ∈ (0.5,1) for q_int=0)
    wire [NR_BITS:0] Q_rnd = qi_cv ? Q_corr : {Q_corr[NR_BITS-1:0], 1'b0};
    // Q_rnd[NR_BITS] = 1 always (implicit leading 1)

    // Extract mantissa (24-bit incl. implicit-1), guard, round, sticky
    wire [23:0] man_cv      = Q_rnd[NR_BITS -: 24];   // bits [26:3]
    wire        guard_cv    = Q_rnd[2];
    wire        round_cv    = Q_rnd[1];
    wire        sticky_cv   = Q_rnd[0] | sticky_r;

    // Result biased exponent: subtract 1 when we normalised by ×2 (q_int=0 case)
    wire signed [9:0] exp_res = $signed(exp_cv) - (qi_cv ? 10'sd0 : 10'sd1);

    // =========================================================================
    // CONV pipeline register (latency cycle 15)
    // =========================================================================
    // Packing: {man[23:0], guard, round, sticky, exp[9:0], sign, frm[2:0], exc[4:0]}
    localparam CONV_W = 24 + 1 + 1 + 1 + 10 + 1 + INST_FRM_BITS + 5; // = 46

    wire [CONV_W-1:0] conv_in =
        {man_cv, guard_cv, round_cv, sticky_cv, exp_res, sgn_cv, frm_cv, exc_cv};

    wire [CONV_W-1:0] conv_out;
    VX_pipe_register #(
        .DATAW (CONV_W),
        .DEPTH (1)
    ) conv_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in (conv_in),
        .data_out(conv_out)
    );

    // =========================================================================
    // NRM stage (combinational): round, overflow/underflow, pack result
    // =========================================================================
    // Unpack CONV output
    wire [23:0]              s_man;
    wire                     s_guard, s_round, s_sticky;
    wire [9:0]               s_exp_bits;
    wire                     s_sign;
    wire [INST_FRM_BITS-1:0] s_frm;
    wire [4:0]               s_exc;

    assign {s_man, s_guard, s_round, s_sticky, s_exp_bits, s_sign, s_frm, s_exc} = conv_out;

    wire signed [9:0] s_exp = $signed(s_exp_bits);

    // RISC-V rounding (sign-magnitude on 24-bit mantissa incl. implicit-1)
    wire [23:0] abs_rounded;
    wire        round_sign;
    wire        exact_zero;

    VX_fp_rounding #(.DAT_WIDTH(24)) u_rnd (
        .abs_value_i           (s_man),
        .sign_i                (s_sign),
        .round_sticky_bits_i   ({s_guard, s_round | s_sticky}),
        .rnd_mode_i            (s_frm),
        .effective_subtraction_i(1'b0),
        .abs_rounded_o         (abs_rounded),
        .sign_o                (round_sign),
        .exact_zero_o          (exact_zero)
    );

    // Rounding overflow: 24'hFFFFFF + 1 wraps to 0 → bump exponent
    wire round_carry   = (abs_rounded == '0) & (s_man != '0);
    wire [22:0] fin_man = round_carry ? 23'd0 : abs_rounded[22:0];
    wire signed [9:0] fin_exp = s_exp + (round_carry ? 10'sd1 : 10'sd0);

    // Exception flags
    wire is_nan  = s_exc[4];
    wire is_inf  = s_exc[3];
    wire is_zero = s_exc[2];
    wire dz_flag = s_exc[1];
    wire nv_flag = s_exc[0];

    wire of_flag = (fin_exp >= 10'sd255) & ~is_nan & ~is_inf;
    wire uf_flag = (fin_exp <= 10'sd0)   & ~is_nan & ~is_inf & ~is_zero & ~exact_zero;
    wire nx_flag = (s_guard | s_round | s_sticky) & ~is_nan & ~is_inf;

    logic [31:0] nrm_result;
    always_comb begin
        if (is_nan) begin
            nrm_result = 32'h7FC00000;             // canonical quiet NaN
        end else if (is_inf | of_flag) begin
            nrm_result = {round_sign, 8'hFF, 23'd0}; // ±Inf
        end else if (is_zero | exact_zero | uf_flag) begin
            nrm_result = {round_sign, 31'd0};         // ±0 (flush-to-zero on underflow)
        end else begin
            nrm_result = {round_sign, fin_exp[7:0], fin_man};
        end
    end

    fflags_t nrm_fflags;
    assign nrm_fflags.NV = nv_flag;
    assign nrm_fflags.DZ = dz_flag;
    assign nrm_fflags.OF = of_flag;
    assign nrm_fflags.UF = uf_flag;
    assign nrm_fflags.NX = nx_flag | of_flag | uf_flag;

    // =========================================================================
    // NRM pipeline register (latency cycle 16 = LATENCY_FDIV)
    // =========================================================================
    VX_pipe_register #(
        .DATAW (32 + `FP_FLAGS_BITS),
        .DEPTH (1)
    ) nrm_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in ({nrm_result, nrm_fflags}),
        .data_out({result,     fflags})
    );

endmodule

`endif
