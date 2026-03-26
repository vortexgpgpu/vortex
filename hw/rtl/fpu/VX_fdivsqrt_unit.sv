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

// Single-lane pipelined fused F32 FDIV + FSQRT.
// Pipeline: 1 PRE + 13 SRT + 1 CONV + 1 NRM = 16 cycles (LATENCY default).
// Supports all RISC-V rounding modes. Produces fflags: NV, DZ, OF, UF, NX.
//
// DIV: Non-restoring radix-2 NR algorithm identical to VX_fdiv_unit.
//   W, D scaled ×32; D ∈ [2^28, 2^29); NR invariant |W| ≤ D.
//
// SQRT: NR radix-2 with W and S (partial quotient) scaled ×16.
//   S ∈ [2^27, 2^28); D_eff = 2S + ulp_k ∈ [2^28, 2^29) — same range as D_div.
//   Q ∈ [1,2) always (q_int=1); for odd biased-exp inputs, sig is scaled ×2
//   before the SRT loop and Q_0=1.5 (nr_offset=1) is used when sig ≥ 1.125
//   to maintain |W_0| ≤ D_eff_0.
//   NR→binary CONV: S_final = Q*2^27 directly encodes man/G/R/S bits.
//
// NR SQRT step k (two per stage):
//   D_eff = 2*S + ulp_k  (computed from current S and compile-time ulp)
//   W_next = 2*W ∓ D_eff  (sign chosen by sign of W, same structure as DIV)
//   S_next = S ± ulp_k    (update partial quotient)
//
// State vector (108 bits):
//   [107]     is_sqrt
//   [106]     nr_offset  (SQRT: 1 when Q_0=1.5, i.e. odd-biased-exp + sig≥1.125)
//   [105:76]  W[29:0]
//   [75:46]   DS[29:0]   (D for DIV, S for SQRT)
//   [45:20]   q_bits[25:0]
//   [19]      q_int
//   [18:9]    exp_r[9:0]
//   [8]       sign_r
//   [7:5]     frm[2:0]
//   [4:0]     exc[4:0]

`include "VX_fpu_define.vh"

module VX_fdivsqrt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY = 16
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,

    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [31:0] dataa,   // dividend (DIV) or radicand (SQRT)
    input  wire [31:0] datab,   // divisor  (DIV only; tie to 0 for SQRT)
    input  wire        is_sqrt,

    output wire [31:0]               result,
    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    `UNUSED_VAR (fmt)   // F32-only

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam PRE_LATENCY  = 1;
    localparam CONV_LATENCY = 1;
    localparam NRM_LATENCY  = 1;
    localparam MAN_BITS    = 23;
    localparam EXP_BITS    = 8;
    localparam SIG_BITS    = MAN_BITS + 1;          // 24
    localparam SUBLZC_BITS = `LOG2UP(SIG_BITS);     // 5
    localparam W_BITS      = 30;
    localparam NR_BITS     = 26;
    localparam NR_STAGES   = NR_BITS / 2;           // 13
    `STATIC_ASSERT(LATENCY == (PRE_LATENCY + NR_STAGES + CONV_LATENCY + NRM_LATENCY), ("VX_fdivsqrt_unit: LATENCY must be %0d, got %0d", PRE_LATENCY+NR_STAGES+CONV_LATENCY+NRM_LATENCY, LATENCY))

    localparam EXC_LO  = 0;
    localparam FRM_LO  = EXC_LO  + 5;
    localparam SGN_LO  = FRM_LO  + INST_FRM_BITS;  // 8
    localparam EXP_LO  = SGN_LO  + 1;              // 9
    localparam QI_LO   = EXP_LO  + 10;             // 19
    localparam QB_LO   = QI_LO   + 1;              // 20
    localparam DS_LO   = QB_LO   + NR_BITS;        // 46  (D for DIV, S for SQRT)
    localparam W_LO    = DS_LO   + W_BITS;         // 76
    localparam NRO_LO  = W_LO    + W_BITS;         // 106 (nr_offset)
    localparam SQ_LO   = NRO_LO  + 1;             // 107 (is_sqrt)
    localparam STAGE_W = SQ_LO   + 1;             // 108

    // =========================================================================
    // Input classification and LZC normalization (shared for DIV and SQRT)
    // =========================================================================
    wire        s_a = dataa[31],  s_b = datab[31];
    wire [7:0]  e_a = dataa[30:23], e_b = datab[30:23];
    wire [22:0] m_a = dataa[22:0],  m_b = datab[22:0];

    fclass_t clss_a, clss_b;
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_a (
        .exp_i(e_a), .man_i(m_a), .clss_o(clss_a));
    VX_fp_classifier #(.MAN_BITS(MAN_BITS), .EXP_BITS(EXP_BITS)) cls_b (
        .exp_i(e_b), .man_i(m_b), .clss_o(clss_b));
    `UNUSED_VAR ({clss_a.is_normal, clss_a.is_quiet,
                  clss_b.is_normal, clss_b.is_quiet})

    wire [SIG_BITS-1:0] sig_a_raw = {~(clss_a.is_zero | clss_a.is_subnormal), m_a};
    wire [SIG_BITS-1:0] sig_b_raw = {~(clss_b.is_zero | clss_b.is_subnormal), m_b};

    wire [SUBLZC_BITS-1:0] lzc_a, lzc_b;
    wire lzc_a_vld, lzc_b_vld;
    VX_lzc #(.N(SIG_BITS)) lzc_suba (.data_in(sig_a_raw), .data_out(lzc_a), .valid_out(lzc_a_vld));
    VX_lzc #(.N(SIG_BITS)) lzc_subb (.data_in(sig_b_raw), .data_out(lzc_b), .valid_out(lzc_b_vld));
    `UNUSED_VAR ({lzc_a_vld, lzc_b_vld})

    wire [SIG_BITS-1:0] sig_a_norm = clss_a.is_subnormal ? (sig_a_raw << lzc_a) : sig_a_raw;
    wire [SIG_BITS-1:0] sig_b_norm = clss_b.is_subnormal ? (sig_b_raw << lzc_b) : sig_b_raw;

    wire signed [9:0] exp_a = clss_a.is_subnormal ? (10'sd1 - 10'(lzc_a))
                            : clss_a.is_zero       ? 10'sd0
                            :                        10'(e_a);
    wire signed [9:0] exp_b = clss_b.is_subnormal ? (10'sd1 - 10'(lzc_b))
                            : clss_b.is_zero       ? 10'sd0
                            :                        10'(e_b);

    // =========================================================================
    // PRE: DIV path
    // =========================================================================
    wire        sign_r0_div = s_a ^ s_b;
    wire signed [9:0] exp_r0_div = exp_a - exp_b + 10'sd127;

    wire nan_a  = clss_a.is_nan,    nan_b  = clss_b.is_nan;
    wire inf_a  = clss_a.is_inf,    inf_b  = clss_b.is_inf;
    wire zero_a = clss_a.is_zero,   zero_b = clss_b.is_zero;
    wire snan_a = clss_a.is_signaling, snan_b = clss_b.is_signaling;

    wire nv0_div  = snan_a | snan_b | (zero_a & zero_b) | (inf_a & inf_b);
    wire dz0_div  = zero_b & ~nan_a & ~nan_b & ~zero_a;
    wire rnan_div = nan_a | nan_b | nv0_div;
    wire rinf_div = (inf_a | zero_b) & ~rnan_div;
    wire rzro_div = (zero_a | inf_b) & ~rnan_div;
    wire [4:0] exc0_div = {rnan_div, rinf_div, rzro_div, dz0_div, nv0_div};

    // Divisor D0 = sig_b_norm * 32 ∈ [2^28, 2^29); W0 = (sig_a - q_int*sig_b)*32 ∈ [0, D0)
    wire [W_BITS-1:0]   D0_div      = {1'b0, sig_b_norm, 5'b0};
    wire                q_int0_div  = (sig_a_norm >= sig_b_norm);
    wire [SIG_BITS-1:0] pre_diff    = sig_a_norm - (q_int0_div ? sig_b_norm : {SIG_BITS{1'b0}});
    wire [W_BITS-1:0]   W0_div      = {1'b0, pre_diff, 5'b0};

    // =========================================================================
    // PRE: SQRT path
    // =========================================================================
    // Result biased exponent: floor((e_a + 127) / 2).
    // Works for both even/odd biased exponents and for subnormal inputs after LZC.
    wire signed [9:0] exp_r0_sqrt = ($signed(exp_a) + 10'sd127) >>> 1;

    // Special cases for SQRT
    wire nv0_sq   = clss_a.is_signaling | (s_a & ~clss_a.is_nan & ~clss_a.is_zero);
    wire rnan_sq  = clss_a.is_nan | nv0_sq;
    wire rinf_sq  = clss_a.is_inf & ~rnan_sq;
    wire rzro_sq  = clss_a.is_zero & ~rnan_sq;
    wire [4:0] exc0_sq  = {rnan_sq, rinf_sq, rzro_sq, 1'b0, nv0_sq};
    wire sign_r0_sq = clss_a.is_zero & s_a;  // sqrt(-0) = -0; all other valid: +

    // Biased exponent even → unbiased exp is odd → scale significand ×2 before loop
    // (This makes F = 2*sig ∈ [2,4) with Q ∈ [√2, 2))
    wire is_scale2_sq = ~e_a[0];

    // When scaling, pre-commit first NR bit if sig ≥ 1.125 (i.e., F ≥ 2.25, Q_0=1.5).
    // sig_a_norm ≥ {1'b1, 23'd(2^20)} iff m_a[22:20] ≠ 0.
    wire nr_offset0_sq = is_scale2_sq & (m_a[22] | m_a[21] | m_a[20]);

    // SQRT W_0 and S_0 (scale ×16, so S ∈ [2^27, 2^28), D_eff ∈ [2^28, 2^29)):
    //   Case 1 (no scale, Q_0=1.0):
    //     S_0 = 2^27,   W_0 = (sig-1)*2^27 = m_a*16
    //   Case 2 (scale, q_nr_0=0, Q_0=1.0, F∈[2,2.25)):
    //     S_0 = 2^27,   W_0 = (2*sig-1)*2^27 = (2^23 + 2*m_a)*16 = 2^27 + m_a*32
    //   Case 3 (scale, q_nr_0=1, Q_0=1.5, F∈[2.25,4)):
    //     S_0 = 3*2^26, W_0 = (2*sig-2.25)*2^27 = m_a*32 - 2^25
    wire [W_BITS-1:0] ma32 = W_BITS'(m_a) << 5;  // m_a * 32 (30-bit, always < 2^28)

    wire [W_BITS-1:0] S0_sq = nr_offset0_sq ? W_BITS'(3 << 26)  // 3*2^26
                                             : W_BITS'(1 << 27); // 2^27

    wire [W_BITS-1:0] W0_sq = !is_scale2_sq  ? (W_BITS'(m_a) << 4)         // Case 1
                             : !nr_offset0_sq ? (W_BITS'(1 << 27) + ma32)   // Case 2
                             :                  (ma32 - W_BITS'(1 << 25));   // Case 3

    // =========================================================================
    // PRE pipeline register (latency cycle 1)
    // =========================================================================
    wire [STAGE_W-1:0] srt_stage [0:NR_STAGES];
    wire [STAGE_W-1:0] pre_in;

    assign pre_in[W_LO  +: W_BITS]         = is_sqrt ? W0_sq               : W0_div;
    assign pre_in[DS_LO +: W_BITS]         = is_sqrt ? S0_sq               : D0_div;
    assign pre_in[QB_LO +: NR_BITS]        = {NR_BITS{1'b0}};
    assign pre_in[QI_LO]                   = is_sqrt ? 1'b1                 : q_int0_div; // SQRT: q_int=1 always
    assign pre_in[EXP_LO +: 10]            = is_sqrt ? 10'(exp_r0_sqrt)     : 10'(exp_r0_div);
    assign pre_in[SGN_LO]                  = is_sqrt ? sign_r0_sq           : sign_r0_div;
    assign pre_in[FRM_LO +: INST_FRM_BITS] = frm;
    assign pre_in[EXC_LO +: 5]             = is_sqrt ? exc0_sq              : exc0_div;
    assign pre_in[NRO_LO]                  = is_sqrt ? nr_offset0_sq        : 1'b0;
    assign pre_in[SQ_LO]                   = is_sqrt;

    VX_pipe_register #(.DATAW(STAGE_W), .DEPTH(1)) pre_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in(pre_in), .data_out(srt_stage[0])
    );

    // =========================================================================
    // SRT stages 1..13 (latency cycles 2..14)
    // Each stage performs two NR steps (A and B).
    //
    // DIV step:        W = 2W ∓ D  (∓: subtract when W≥0, add when W<0)
    // SQRT step NRO=0: W = (q ? 2W-2S : 2W+2S) - ulp;  S = S ± ulp
    // SQRT step NRO=1: W = (q ? 2W- S : 2W+ S) - ulp/2; S = S ± ulp
    //
    // SQRT ULP schedule (stage k ∈ [0,12]):
    //   nr_offset=0: ulp_A = 2^(26-2k), ulp_B = 2^(25-2k)
    //   nr_offset=1: ulp_A = 2^(25-2k), ulp_B = 2^(24-2k)  (one position earlier)
    // =========================================================================
    for (genvar k = 0; k < NR_STAGES; k++) begin : g_srt
        // Compile-time ULP constants (no runtime cost, just mux between two constants)
        localparam [W_BITS-1:0] ULP_A_NRO0 = W_BITS'(1) << (26 - 2*k);
        localparam [W_BITS-1:0] ULP_B_NRO0 = W_BITS'(1) << (25 - 2*k);
        localparam [W_BITS-1:0] ULP_A_NRO1 = W_BITS'(1) << (25 - 2*k);
        localparam [W_BITS-1:0] ULP_B_NRO1 = W_BITS'(1) << (24 - 2*k);

        // Unpack state
        wire [W_BITS-1:0]        W_in   = srt_stage[k][W_LO  +: W_BITS];
        wire [W_BITS-1:0]        DS_in  = srt_stage[k][DS_LO +: W_BITS];
        wire [NR_BITS-1:0]       qb_in  = srt_stage[k][QB_LO +: NR_BITS];
        wire                     qi_in  = srt_stage[k][QI_LO];
        wire [9:0]               exp_in = srt_stage[k][EXP_LO +: 10];
        wire                     sgn_in = srt_stage[k][SGN_LO];
        wire [INST_FRM_BITS-1:0] frm_in = srt_stage[k][FRM_LO +: INST_FRM_BITS];
        wire [4:0]               exc_in = srt_stage[k][EXC_LO +: 5];
        wire                     nro_in = srt_stage[k][NRO_LO];
        wire                     sq_in  = srt_stage[k][SQ_LO];

        // Runtime ULP selection (2-way mux on compile-time constants)
        wire [W_BITS-1:0] ulp_a = nro_in ? ULP_A_NRO1 : ULP_A_NRO0;
        wire [W_BITS-1:0] ulp_b = nro_in ? ULP_B_NRO1 : ULP_B_NRO0;

        // --- NR Step A ---
        // DIV:        W = 2W ∓ D      (∓: subtract when q=1, add when q=0)
        // SQRT NRO=0: W = (q ? 2W-2S : 2W+2S) - ulp_a
        // SQRT NRO=1: W = (q ? 2W- S : 2W+ S) - ulp_a/2
        wire [W_BITS:0] D_S_a   = sq_in ? (nro_in ? {1'b0, DS_in} : {DS_in, 1'b0})
                                        : {1'b0, DS_in};
        wire [W_BITS:0] D_ulp_a = sq_in ? {1'b0, nro_in ? (ulp_a >> 1) : ulp_a}
                                        : (W_BITS+1)'(0);
        wire q_a = ~W_in[W_BITS-1];                      // 1 iff W_in ≥ 0
        wire [W_BITS:0] Wa_pm = q_a ? ({W_in, 1'b0} - D_S_a) : ({W_in, 1'b0} + D_S_a);
        wire [W_BITS:0] Wa31  = Wa_pm - D_ulp_a;
        `UNUSED_VAR (Wa31[W_BITS])                        // invariant guarantees carry is valid
        wire [W_BITS-1:0] W_a = Wa31[W_BITS-1:0];

        // DS update: DIV → unchanged; SQRT → S ± ulp_a
        wire [W_BITS-1:0] DS_a = sq_in ? (q_a ? DS_in + ulp_a : DS_in - ulp_a)
                                       : DS_in;

        // --- NR Step B ---
        wire [W_BITS:0] D_S_b   = sq_in ? (nro_in ? {1'b0, DS_a} : {DS_a, 1'b0})
                                        : {1'b0, DS_a};
        wire [W_BITS:0] D_ulp_b = sq_in ? {1'b0, nro_in ? (ulp_b >> 1) : ulp_b}
                                        : (W_BITS+1)'(0);
        wire q_b = ~W_a[W_BITS-1];
        wire [W_BITS:0] Wb_pm = q_b ? ({W_a, 1'b0} - D_S_b) : ({W_a, 1'b0} + D_S_b);
        wire [W_BITS:0] Wb31  = Wb_pm - D_ulp_b;
        `UNUSED_VAR (Wb31[W_BITS])
        wire [W_BITS-1:0] W_b = Wb31[W_BITS-1:0];

        wire [W_BITS-1:0] DS_b = sq_in ? (q_b ? DS_a + ulp_b : DS_a - ulp_b)
                                       : DS_a;

        // Accumulate NR bits for DIV CONV (shifted in LSB-first; MSBs fall off)
        `UNUSED_VAR (qb_in[NR_BITS-1:NR_BITS-2])
        wire [NR_BITS-1:0] qb_new = {qb_in[NR_BITS-3:0], q_a, q_b};

        // Pack output state
        wire [STAGE_W-1:0] s_out;
        assign s_out[W_LO  +: W_BITS]         = W_b;
        assign s_out[DS_LO +: W_BITS]         = DS_b;
        assign s_out[QB_LO +: NR_BITS]        = qb_new;
        assign s_out[QI_LO]                   = qi_in;
        assign s_out[EXP_LO +: 10]            = exp_in;
        assign s_out[SGN_LO]                  = sgn_in;
        assign s_out[FRM_LO +: INST_FRM_BITS] = frm_in;
        assign s_out[EXC_LO +: 5]             = exc_in;
        assign s_out[NRO_LO]                  = nro_in;
        assign s_out[SQ_LO]                   = sq_in;

        VX_pipe_register #(.DATAW(STAGE_W), .DEPTH(1)) srt_reg (
            .clk(clk), .reset(reset), .enable(enable),
            .data_in(s_out), .data_out(srt_stage[k+1])
        );
    end

    // =========================================================================
    // CONV (combinational after srt_stage[NR_STAGES])
    // =========================================================================
    wire [W_BITS-1:0]        W_cv   = srt_stage[NR_STAGES][W_LO  +: W_BITS];
    wire [W_BITS-1:0]        DS_cv  = srt_stage[NR_STAGES][DS_LO +: W_BITS];
    wire [NR_BITS-1:0]       qb_cv  = srt_stage[NR_STAGES][QB_LO +: NR_BITS];
    wire                     qi_cv  = srt_stage[NR_STAGES][QI_LO];
    wire [9:0]               exp_cv = srt_stage[NR_STAGES][EXP_LO +: 10];
    wire                     sgn_cv = srt_stage[NR_STAGES][SGN_LO];
    wire [INST_FRM_BITS-1:0] frm_cv = srt_stage[NR_STAGES][FRM_LO +: INST_FRM_BITS];
    wire [4:0]               exc_cv = srt_stage[NR_STAGES][EXC_LO +: 5];
    wire                     nro_cv = srt_stage[NR_STAGES][NRO_LO];
    wire                     sq_cv  = srt_stage[NR_STAGES][SQ_LO];

    // ---- DIV CONV: NR → binary (same as VX_fdiv_unit) ----
    wire [NR_BITS:0] Q_frac = {qb_cv, 1'b0} - {1'b0, {NR_BITS{1'b1}}}; // 2*qb - (2^26-1)
    `UNUSED_VAR (Q_frac[NR_BITS])
    wire [NR_BITS:0] Q_tot  = {qi_cv, Q_frac[NR_BITS-1:0]};

    wire              W_neg_div = W_cv[W_BITS-1];
    wire [NR_BITS:0]  Q_corr    = W_neg_div ? (Q_tot - 1'b1) : Q_tot;
    wire [W_BITS-1:0] W_corr_div = W_neg_div ? (W_cv + DS_cv) : W_cv;
    wire              sticky_div_r = (W_corr_div != '0);

    wire [NR_BITS:0] Q_rnd = qi_cv ? Q_corr : {Q_corr[NR_BITS-1:0], 1'b0};
    wire [23:0] man_div  = Q_rnd[NR_BITS -: 24];   // bits [26:3]
    wire guard_div       = Q_rnd[2];
    wire round_div       = Q_rnd[1];
    wire sticky_div      = Q_rnd[0] | sticky_div_r;
    wire signed [9:0] exp_res_div = $signed(exp_cv) - (qi_cv ? 10'sd0 : 10'sd1);

    // ---- SQRT CONV: DS_cv = S_final = Q*2^27 ∈ [2^27, 2^28) ----
    // NR correction: if W_final < 0, Q was over-estimated by one ULP.
    //   ULP_last: nr_offset=0 → 2 (step-B ulp of last stage),
    //             nr_offset=1 → 1
    //   S_corr = DS_cv - ULP_last
    //   W_corr = W_final + DS_cv - ULP_last        (NRO=1: undo W = 2W - S_prev,  S_prev = S_cv - 1)
    //          = W_final + 2*DS_cv - ULP_last       (NRO=0: undo W = 2W - 2*S_prev - ULP, S_prev = S_cv - 2)
    wire              W_neg_sq  = W_cv[W_BITS-1];
    wire [W_BITS-1:0] ulp_last  = nro_cv ? W_BITS'(1) : W_BITS'(2);
    wire [W_BITS-1:0] S_corr    = W_neg_sq ? (DS_cv - ulp_last) : DS_cv;
    `UNUSED_VAR (S_corr[W_BITS-1:28])  // S ∈ [2^27, 2^28), upper bits always 0

    wire [W_BITS:0]   W_cv_sx   = {W_cv[W_BITS-1], W_cv};  // sign-extend W_final to 31 bits
    wire [W_BITS:0]   W_D_sq    = nro_cv ? ({1'b0, DS_cv} - {1'b0, ulp_last})
                                         : ({DS_cv, 1'b0}  - {1'b0, ulp_last});
    wire [W_BITS:0]   W_corr_sq31 = W_neg_sq ? (W_cv_sx + W_D_sq) : W_cv_sx;
    `UNUSED_VAR (W_corr_sq31[W_BITS])

    wire sticky_sq_w  = (W_corr_sq31[W_BITS-1:0] != '0); // nonzero corrected remainder
    wire sticky_sq_lo = (S_corr[1:0] != 2'b00);          // sub-round bits from S_corr

    // S_corr ∈ [2^27, 2^28): bit 27 = implicit-1, bits 26:4 = mantissa, bits 3:0 = G/R/S
    wire [23:0] man_sq    = S_corr[27:4];
    wire        guard_sq  = S_corr[3];
    wire        round_sq  = S_corr[2];
    wire        sticky_sq = sticky_sq_lo | sticky_sq_w;
    // SQRT exp unchanged: Q ∈ [1,2) always (q_int=1 was stored in PRE)
    wire signed [9:0] exp_res_sq = $signed(exp_cv);

    // ---- Unified CONV output ----
    wire [23:0]       man_cv_out    = sq_cv ? man_sq    : man_div;
    wire              guard_cv_out  = sq_cv ? guard_sq  : guard_div;
    wire              round_cv_out  = sq_cv ? round_sq  : round_div;
    wire              sticky_cv_out = sq_cv ? sticky_sq : sticky_div;
    wire signed [9:0] exp_res       = sq_cv ? exp_res_sq : exp_res_div;

    // =========================================================================
    // CONV pipeline register (latency cycle 15)
    // =========================================================================
    localparam CONV_W = 24 + 1 + 1 + 1 + 10 + 1 + INST_FRM_BITS + 5;  // 46

    wire [CONV_W-1:0] conv_in = {man_cv_out, guard_cv_out, round_cv_out, sticky_cv_out,
                                  exp_res, sgn_cv, frm_cv, exc_cv};
    wire [CONV_W-1:0] conv_out;
    VX_pipe_register #(.DATAW(CONV_W), .DEPTH(1)) conv_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in(conv_in), .data_out(conv_out)
    );

    // =========================================================================
    // NRM stage (combinational): round, overflow/underflow, pack result
    // =========================================================================
    wire [23:0]              s_man;
    wire                     s_guard, s_round, s_sticky;
    wire [9:0]               s_exp_bits;
    wire                     s_sign;
    wire [INST_FRM_BITS-1:0] s_frm;
    wire [4:0]               s_exc;

    assign {s_man, s_guard, s_round, s_sticky, s_exp_bits, s_sign, s_frm, s_exc} = conv_out;
    wire signed [9:0] s_exp = $signed(s_exp_bits);

    wire [23:0] abs_rounded;
    wire        round_sign, exact_zero;
    VX_fp_rounding #(.DAT_WIDTH(24)) u_rnd (
        .abs_value_i            (s_man),
        .sign_i                 (s_sign),
        .round_sticky_bits_i    ({s_guard, s_round | s_sticky}),
        .rnd_mode_i             (s_frm),
        .effective_subtraction_i(1'b0),
        .abs_rounded_o          (abs_rounded),
        .sign_o                 (round_sign),
        .exact_zero_o           (exact_zero)
    );

    wire round_carry  = (abs_rounded == '0) & (s_man != '0);
    wire [22:0] fin_man = round_carry ? 23'd0 : abs_rounded[22:0];
    wire signed [9:0] fin_exp = s_exp + (round_carry ? 10'sd1 : 10'sd0);

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
        if (is_nan)                        nrm_result = 32'h7FC00000;
        else if (is_inf | of_flag)         nrm_result = {round_sign, 8'hFF, 23'd0};
        else if (is_zero | exact_zero | uf_flag) nrm_result = {round_sign, 31'd0};
        else                               nrm_result = {round_sign, fin_exp[7:0], fin_man};
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
    VX_pipe_register #(.DATAW(32 + `FP_FLAGS_BITS), .DEPTH(1)) nrm_reg (
        .clk(clk), .reset(reset), .enable(enable),
        .data_in ({nrm_result, nrm_fflags}),
        .data_out({result,     fflags})
    );

endmodule
