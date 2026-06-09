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

// Merged single-lane pipelined fused FDIV + FSQRT.
//
// Merged-format design: the iterative SRT datapath is sized for the widest
// supported format (SUPER = F64 when FLEN>=64, else F32) and runs the SAME
// carry-save NR recurrence regardless of the operation's format. Only the
// front (unpack) and back (extract/round/pack) stages are format-aware:
//   - operands are unpacked at their active format (F32/F64) and the
//     normalized significand is LEFT-justified into the SUPER significand,
//   - the exponent is tracked in the ACTIVE format's frame (bias/range), so
//     an F32 op rounds ONCE at 24-bit precision (no double rounding) and
//     under/overflows at the F32 boundary.
//
// Pipeline: 1 PRE + 1 INI + NR_STAGES SRT + 1 CONV + 1 NRM.
//   SUPER=F32: NR_STAGES=13 -> LATENCY 17 (bit-identical to the legacy F32 unit)
//   SUPER=F64: NR_STAGES=28 -> LATENCY 32
//
// DIV: Non-restoring radix-2 NR.  W, D scaled x2^SCALE_DIV; NR invariant |W|<=D.
// SQRT: NR radix-2 with W and S (partial root) scaled x2^SCALE_SQRT.
//   SCALE_SQRT is chosen by SUPER_SIG parity so the per-stage ULP schedule
//   lands on the same endpoints (step-B ULP last = 2 for NRO=0, 1 for NRO=1)
//   for both formats, keeping the CONV correction format-independent.

`include "VX_fpu_define.vh"

module VX_fdivsqrt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY = 17,
    parameter FLEN    = 32
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire mask,

    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [FLEN-1:0] dataa,   // dividend (DIV) or radicand (SQRT)
    input  wire [FLEN-1:0] datab,   // divisor  (DIV only; ignored for SQRT)
    input  wire        is_sqrt,

    output wire [FLEN-1:0] result,
    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam HAS_D     = (FLEN >= 64);
    wire is_d = HAS_D ? fmt[0] : 1'b0;
    `UNUSED_VAR (fmt)

    // ---- SUPER (widest) format the datapath is built for ----
    localparam SUPER_MAN = HAS_D ? 52 : 23;
    localparam SUPER_SIG = SUPER_MAN + 1;                 // 24 / 53

    localparam PRE_LATENCY  = 1;
    localparam INI_LATENCY  = 1;
    localparam CONV_LATENCY = 1;
    localparam NRM_LATENCY  = 1;

    localparam SCALE_DIV  = 5;
    localparam SCALE_SQRT = (SUPER_SIG % 2 == 1) ? 5 : 4; // even SIG->4 (F32), odd->5 (F64)

    localparam W_BITS   = SUPER_SIG + SCALE_DIV + 1;      // 30 / 59
    localparam CS_BITS  = W_BITS + 1;                     // 31 / 60: carry-save component width
    localparam NR_BITS  = (SUPER_SIG + 2) + ((SUPER_SIG + 2) % 2); // even: 26 / 56
    localparam NR_STAGES = NR_BITS / 2;                  // 13 / 28
    localparam EXP_W    = HAS_D ? 14 : 10;               // signed exponent working width

    // SQRT scaled-S leading-bit position: S in [2^SQRT_LEAD, 2^(SQRT_LEAD+1))
    localparam SQRT_LEAD = SUPER_SIG + SCALE_SQRT - 1;   // 27 / 57

    `STATIC_ASSERT(LATENCY == (PRE_LATENCY + INI_LATENCY + NR_STAGES + CONV_LATENCY + NRM_LATENCY), ("VX_fdivsqrt_unit: LATENCY must be %0d, got %0d", PRE_LATENCY+INI_LATENCY+NR_STAGES+CONV_LATENCY+NRM_LATENCY, LATENCY))

    reg [LATENCY-1:0] mask_pipe;
    always @(posedge clk) begin
        if (reset) begin
            mask_pipe <= '0;
        end else if (enable) begin
            mask_pipe <= {mask_pipe[LATENCY-2:0], mask};
        end
    end
    wire valid_ini  = mask_pipe[PRE_LATENCY-1];
    wire valid_conv = mask_pipe[PRE_LATENCY+INI_LATENCY+NR_STAGES-1];
    wire valid_nrm  = mask_pipe[PRE_LATENCY+INI_LATENCY+NR_STAGES+CONV_LATENCY-1];

    // ---- SRT state vector layout ----
    localparam EXC_LO  = 0;
    localparam FRM_LO  = EXC_LO  + 5;
    localparam SGN_LO  = FRM_LO  + INST_FRM_BITS;
    localparam EXP_LO  = SGN_LO  + 1;
    localparam QI_LO   = EXP_LO  + EXP_W;
    localparam QB_LO   = QI_LO   + 1;
    localparam DS_LO   = QB_LO   + NR_BITS;               // D (DIV) or S (SQRT)
    localparam W_LO    = DS_LO   + W_BITS;                // Ws (CS_BITS wide)
    localparam WC_LO   = W_LO    + CS_BITS;               // Wc (CS_BITS wide)
    localparam NRO_LO  = WC_LO   + CS_BITS;               // nr_offset
    localparam SQ_LO   = NRO_LO  + 1;                     // is_sqrt
    localparam ISD_LO  = SQ_LO   + 1;                     // is_double
    localparam STAGE_W = ISD_LO  + 1;

    // =========================================================================
    // Active-format unpack: classify, LZC-normalize the active significand, then
    // LEFT-justify it into the SUPER significand. Exponent stays in active frame.
    // =========================================================================
    wire [EXP_W-1:0] BIAS = is_d ? EXP_W'(1023) : EXP_W'(127);

    // -- raw active field extraction --
    wire s_a = is_d ? dataa[FLEN-1] : dataa[31];
    wire s_b = is_d ? datab[FLEN-1] : datab[31];

    // active exponent / mantissa selectors and classification
    wire ea_allones, eb_allones, ea_zero, eb_zero, ma_nz, mb_nz, ma_q, mb_q;
    wire [SUPER_SIG-1:0] siga_ljn, sigb_ljn;   // normalized, left-justified into SUPER_SIG
    wire signed [EXP_W-1:0] exp_a, exp_b;

    // Build {classify, normalize, left-justify} for one operand at a chosen format.
    // Implemented inline per format and muxed by is_d (when HAS_D).
    // --- F32 view ---
    wire        f32a_z  = (dataa[30:23] == 8'd0) && (dataa[22:0] == 23'd0);
    wire        f32a_sub= (dataa[30:23] == 8'd0) && (dataa[22:0] != 23'd0);
    wire        f32b_z  = (datab[30:23] == 8'd0) && (datab[22:0] == 23'd0);
    wire        f32b_sub= (datab[30:23] == 8'd0) && (datab[22:0] != 23'd0);
    wire [23:0] f32a_sig = {~(f32a_z | f32a_sub), dataa[22:0]};
    wire [23:0] f32b_sig = {~(f32b_z | f32b_sub), datab[22:0]};
    wire [4:0]  f32a_lzc, f32b_lzc;
    wire f32a_lvld, f32b_lvld;
    VX_lzc #(.N(24)) lz_f32a (.data_in(f32a_sig), .data_out(f32a_lzc), .valid_out(f32a_lvld));
    VX_lzc #(.N(24)) lz_f32b (.data_in(f32b_sig), .data_out(f32b_lzc), .valid_out(f32b_lvld));
    wire [23:0] f32a_norm = f32a_sub ? (f32a_sig << f32a_lzc) : f32a_sig;
    wire [23:0] f32b_norm = f32b_sub ? (f32b_sig << f32b_lzc) : f32b_sig;
    wire signed [EXP_W-1:0] f32a_exp = f32a_sub ? (EXP_W'(1) - EXP_W'(f32a_lzc)) : f32a_z ? '0 : EXP_W'(dataa[30:23]);
    wire signed [EXP_W-1:0] f32b_exp = f32b_sub ? (EXP_W'(1) - EXP_W'(f32b_lzc)) : f32b_z ? '0 : EXP_W'(datab[30:23]);
    `UNUSED_VAR ({f32a_lvld, f32b_lvld})

    if (HAS_D) begin : g_unpack_d
        wire        f64a_z  = (dataa[62:52] == 11'd0) && (dataa[51:0] == 52'd0);
        wire        f64a_sub= (dataa[62:52] == 11'd0) && (dataa[51:0] != 52'd0);
        wire        f64b_z  = (datab[62:52] == 11'd0) && (datab[51:0] == 52'd0);
        wire        f64b_sub= (datab[62:52] == 11'd0) && (datab[51:0] != 52'd0);
        wire [52:0] f64a_sig = {~(f64a_z | f64a_sub), dataa[51:0]};
        wire [52:0] f64b_sig = {~(f64b_z | f64b_sub), datab[51:0]};
        wire [5:0]  f64a_lzc, f64b_lzc;
        wire f64a_lvld, f64b_lvld;
        VX_lzc #(.N(53)) lz_f64a (.data_in(f64a_sig), .data_out(f64a_lzc), .valid_out(f64a_lvld));
        VX_lzc #(.N(53)) lz_f64b (.data_in(f64b_sig), .data_out(f64b_lzc), .valid_out(f64b_lvld));
        wire [52:0] f64a_norm = f64a_sub ? (f64a_sig << f64a_lzc) : f64a_sig;
        wire [52:0] f64b_norm = f64b_sub ? (f64b_sig << f64b_lzc) : f64b_sig;
        wire signed [EXP_W-1:0] f64a_exp = f64a_sub ? (EXP_W'(1) - EXP_W'(f64a_lzc)) : f64a_z ? '0 : EXP_W'(dataa[62:52]);
        wire signed [EXP_W-1:0] f64b_exp = f64b_sub ? (EXP_W'(1) - EXP_W'(f64b_lzc)) : f64b_z ? '0 : EXP_W'(datab[62:52]);
        `UNUSED_VAR ({f64a_lvld, f64b_lvld})

        assign ea_allones = is_d ? (&dataa[62:52]) : (&dataa[30:23]);
        assign eb_allones = is_d ? (&datab[62:52]) : (&datab[30:23]);
        assign ea_zero    = is_d ? f64a_z  : f32a_z;
        assign eb_zero    = is_d ? f64b_z  : f32b_z;
        assign ma_nz      = is_d ? (|dataa[51:0]) : (|dataa[22:0]);
        assign mb_nz      = is_d ? (|datab[51:0]) : (|datab[22:0]);
        assign ma_q       = is_d ? dataa[51] : dataa[22];   // quiet bit (msb of man)
        assign mb_q       = is_d ? datab[51] : datab[22];
        assign siga_ljn   = is_d ? f64a_norm : {f32a_norm, {(SUPER_SIG-24){1'b0}}};
        assign sigb_ljn   = is_d ? f64b_norm : {f32b_norm, {(SUPER_SIG-24){1'b0}}};
        assign exp_a      = is_d ? f64a_exp : f32a_exp;
        assign exp_b      = is_d ? f64b_exp : f32b_exp;
    end else begin : g_unpack_s
        assign ea_allones = &dataa[30:23];
        assign eb_allones = &datab[30:23];
        assign ea_zero    = f32a_z;
        assign eb_zero    = f32b_z;
        assign ma_nz      = |dataa[22:0];
        assign mb_nz      = |datab[22:0];
        assign ma_q       = dataa[22];
        assign mb_q       = datab[22];
        assign siga_ljn   = f32a_norm;
        assign sigb_ljn   = f32b_norm;
        assign exp_a      = f32a_exp;
        assign exp_b      = f32b_exp;
    end

    // classification flags (active format)
    wire nan_a  = ea_allones &  ma_nz;
    wire nan_b  = eb_allones &  mb_nz;
    wire inf_a  = ea_allones & ~ma_nz;
    wire inf_b  = eb_allones & ~mb_nz;
    wire zero_a = ea_zero    & ~ma_nz;
    wire zero_b = eb_zero    & ~mb_nz;
    wire snan_a = nan_a & ~ma_q;
    wire snan_b = nan_b & ~mb_q;

    // =========================================================================
    // PRE: DIV path
    // =========================================================================
    wire        sign_r0_div  = s_a ^ s_b;
    wire signed [EXP_W-1:0] exp_r0_div = exp_a - exp_b + BIAS;

    wire nv0_div  = snan_a | snan_b | (zero_a & zero_b) | (inf_a & inf_b);
    wire dz0_div  = zero_b & ~nan_a & ~nan_b & ~zero_a;
    wire rnan_div = nan_a | nan_b | nv0_div;
    wire rinf_div = (inf_a | zero_b) & ~rnan_div;
    wire rzro_div = (zero_a | inf_b) & ~rnan_div;
    wire [4:0] exc0_div = {rnan_div, rinf_div, rzro_div, dz0_div, nv0_div};

    // =========================================================================
    // PRE: SQRT path
    // =========================================================================
    wire signed [EXP_W-1:0] exp_r0_sqrt = ($signed(exp_a) + BIAS) >>> 1;

    wire nv0_sq   = snan_a | (s_a & ~nan_a & ~zero_a);
    wire rnan_sq  = nan_a | nv0_sq;
    wire rinf_sq  = inf_a & ~rnan_sq;
    wire rzro_sq  = zero_a & ~rnan_sq;
    wire [4:0] exc0_sq  = {rnan_sq, rinf_sq, rzro_sq, 1'b0, nv0_sq};
    wire sign_r0_sq = zero_a & s_a;   // sqrt(-0) = -0; other valid: +

    // Biased exponent even <=> unbiased odd (both F32/F64 biases are odd) -> scale sig x2.
    // Biased-exponent LSB of the radicand (active format); FLEN=32 has only F32.
    wire ea_lsb_sq;
    if (HAS_D) begin : g_ealsb_d
        assign ea_lsb_sq = is_d ? dataa[52] : dataa[23];
    end else begin : g_ealsb_s
        assign ea_lsb_sq = dataa[23];
    end
    wire is_scale2_sq = ~ea_lsb_sq;
    // Q_0=1.5 pre-commit when scaling and sig >= 1.125 (top 3 mantissa bits nonzero).
    wire [2:0] top3_man = siga_ljn[SUPER_SIG-2 -: 3];
    wire nr_offset0_sq = is_scale2_sq & (|top3_man);

    // man (fraction) as integer at SUPER width
    wire [SUPER_MAN-1:0] man_a = siga_ljn[SUPER_MAN-1:0];

    // SQRT W_0 / S_0 (see header; derived symbolically, reduce to legacy F32 consts)
    wire [W_BITS-1:0] manSC1 = W_BITS'(man_a) << (SCALE_SQRT + 1);  // man * 2^(SC+1)

    wire [W_BITS-1:0] S0_sq = nr_offset0_sq ? W_BITS'(3) << (SQRT_LEAD - 1)  // 3*2^(LEAD-1)
                                            : W_BITS'(1) << SQRT_LEAD;        // 2^LEAD

    wire [W_BITS-1:0] W0_sq = !is_scale2_sq  ? (W_BITS'(man_a) << SCALE_SQRT)              // case1: man*2^SC
                             : !nr_offset0_sq ? ((W_BITS'(1) << SQRT_LEAD) + manSC1)        // case2: 2^LEAD + man*2^(SC+1)
                             :                  (manSC1 - (W_BITS'(1) << (SUPER_SIG + SCALE_SQRT - 3))); // case3

    // =========================================================================
    // PRE pipeline register (cycle 1): normalize + exc.
    // =========================================================================
    localparam INI_SIG_LO   = 0;
    localparam INI_SIGB_LO  = INI_SIG_LO   + SUPER_SIG;
    localparam INI_EXPD_LO  = INI_SIGB_LO  + SUPER_SIG;
    localparam INI_EXPS_LO  = INI_EXPD_LO  + EXP_W;
    localparam INI_SGND_LO  = INI_EXPS_LO  + EXP_W;
    localparam INI_SGNS_LO  = INI_SGND_LO  + 1;
    localparam INI_NRO_LO   = INI_SGNS_LO  + 1;
    localparam INI_W0SQ_LO  = INI_NRO_LO   + 1;
    localparam INI_S0SQ_LO  = INI_W0SQ_LO  + W_BITS;
    localparam INI_EXCDV_LO = INI_S0SQ_LO  + W_BITS;
    localparam INI_EXCSQ_LO = INI_EXCDV_LO + 5;
    localparam INI_FRM_LO   = INI_EXCSQ_LO + 5;
    localparam INI_SQRT_LO  = INI_FRM_LO   + INST_FRM_BITS;
    localparam INI_ISD_LO   = INI_SQRT_LO  + 1;
    localparam INI_W        = INI_ISD_LO   + 1;

    wire [INI_W-1:0] ini_in;
    assign ini_in[INI_SIG_LO   +: SUPER_SIG]     = siga_ljn;
    assign ini_in[INI_SIGB_LO  +: SUPER_SIG]     = sigb_ljn;
    assign ini_in[INI_EXPD_LO  +: EXP_W]         = exp_r0_div;
    assign ini_in[INI_EXPS_LO  +: EXP_W]         = exp_r0_sqrt;
    assign ini_in[INI_SGND_LO]                   = sign_r0_div;
    assign ini_in[INI_SGNS_LO]                   = sign_r0_sq;
    assign ini_in[INI_NRO_LO]                    = nr_offset0_sq;
    assign ini_in[INI_W0SQ_LO  +: W_BITS]        = W0_sq;
    assign ini_in[INI_S0SQ_LO  +: W_BITS]        = S0_sq;
    assign ini_in[INI_EXCDV_LO +: 5]             = exc0_div;
    assign ini_in[INI_EXCSQ_LO +: 5]             = exc0_sq;
    assign ini_in[INI_FRM_LO   +: INST_FRM_BITS] = frm;
    assign ini_in[INI_SQRT_LO]                   = is_sqrt;
    assign ini_in[INI_ISD_LO]                    = is_d;

    wire [INI_W-1:0] ini_out;
    VX_pipe_register #(.DATAW (INI_W), .DEPTH (1)) pre_reg (
        .clk (clk), .reset (reset), .enable (enable && mask),
        .data_in (ini_in), .data_out (ini_out)
    );

    // =========================================================================
    // INI pipeline register (cycle 2): DIV initial W0, pack SRT state.
    // =========================================================================
    wire [SUPER_SIG-1:0]     i_sig_a   = ini_out[INI_SIG_LO   +: SUPER_SIG];
    wire [SUPER_SIG-1:0]     i_sig_b   = ini_out[INI_SIGB_LO  +: SUPER_SIG];
    wire [EXP_W-1:0]         i_exp_div = ini_out[INI_EXPD_LO  +: EXP_W];
    wire [EXP_W-1:0]         i_exp_sq  = ini_out[INI_EXPS_LO  +: EXP_W];
    wire                     i_sgn_div = ini_out[INI_SGND_LO];
    wire                     i_sgn_sq  = ini_out[INI_SGNS_LO];
    wire                     i_nro_sq  = ini_out[INI_NRO_LO];
    wire [W_BITS-1:0]        i_W0_sq   = ini_out[INI_W0SQ_LO  +: W_BITS];
    wire [W_BITS-1:0]        i_S0_sq   = ini_out[INI_S0SQ_LO  +: W_BITS];
    wire [4:0]               i_exc_div = ini_out[INI_EXCDV_LO +: 5];
    wire [4:0]               i_exc_sq  = ini_out[INI_EXCSQ_LO +: 5];
    wire [INST_FRM_BITS-1:0] i_frm     = ini_out[INI_FRM_LO   +: INST_FRM_BITS];
    wire                     i_sqrt    = ini_out[INI_SQRT_LO];
    wire                     i_isd     = ini_out[INI_ISD_LO];

    wire [W_BITS-1:0]    i_D0_div     = {1'b0, i_sig_b, {SCALE_DIV{1'b0}}};
    wire                 i_q_int0_div = (i_sig_a >= i_sig_b);
    wire [SUPER_SIG-1:0] i_pre_diff   = i_sig_a - (i_q_int0_div ? i_sig_b : {SUPER_SIG{1'b0}});
    wire [W_BITS-1:0]    i_W0_div     = {1'b0, i_pre_diff, {SCALE_DIV{1'b0}}};

    wire [STAGE_W-1:0] srt_stage [0:NR_STAGES];
    wire [STAGE_W-1:0] pre_in;

    assign pre_in[W_LO  +: CS_BITS]        = i_sqrt ? {1'b0, i_W0_sq} : {1'b0, i_W0_div};
    assign pre_in[WC_LO +: CS_BITS]        = '0;
    assign pre_in[DS_LO +: W_BITS]         = i_sqrt ? i_S0_sq  : i_D0_div;
    assign pre_in[QB_LO +: NR_BITS]        = {NR_BITS{1'b0}};
    assign pre_in[QI_LO]                   = i_sqrt ? 1'b1     : i_q_int0_div;
    assign pre_in[EXP_LO +: EXP_W]         = i_sqrt ? i_exp_sq : i_exp_div;
    assign pre_in[SGN_LO]                  = i_sqrt ? i_sgn_sq : i_sgn_div;
    assign pre_in[FRM_LO +: INST_FRM_BITS] = i_frm;
    assign pre_in[EXC_LO +: 5]             = i_sqrt ? i_exc_sq : i_exc_div;
    assign pre_in[NRO_LO]                  = i_sqrt ? i_nro_sq : 1'b0;
    assign pre_in[SQ_LO]                   = i_sqrt;
    assign pre_in[ISD_LO]                  = i_isd;

    VX_pipe_register #(.DATAW (STAGE_W), .DEPTH (1)) ini_reg (
        .clk (clk), .reset (reset), .enable (enable && valid_ini),
        .data_in (pre_in), .data_out (srt_stage[0])
    );

    // =========================================================================
    // SRT stages 1..NR_STAGES (two NR steps each, carry-save W).  Format-independent.
    // =========================================================================
    for (genvar k = 0; k < NR_STAGES; k++) begin : g_srt
        localparam [W_BITS-1:0] ULP_A_NRO0 = W_BITS'(1) << (SUPER_SIG + SCALE_SQRT - 2 - 2*k);
        localparam [W_BITS-1:0] ULP_B_NRO0 = W_BITS'(1) << (SUPER_SIG + SCALE_SQRT - 3 - 2*k);
        localparam [W_BITS-1:0] ULP_A_NRO1 = W_BITS'(1) << (SUPER_SIG + SCALE_SQRT - 3 - 2*k);
        localparam [W_BITS-1:0] ULP_B_NRO1 = W_BITS'(1) << (SUPER_SIG + SCALE_SQRT - 4 - 2*k);

        wire [CS_BITS-1:0]       Ws_in  = srt_stage[k][W_LO  +: CS_BITS];
        wire [CS_BITS-1:0]       Wc_in  = srt_stage[k][WC_LO +: CS_BITS];
        wire [W_BITS-1:0]        DS_in  = srt_stage[k][DS_LO +: W_BITS];
        wire [NR_BITS-1:0]       qb_in  = srt_stage[k][QB_LO +: NR_BITS];
        wire                     qi_in  = srt_stage[k][QI_LO];
        wire [EXP_W-1:0]         exp_in = srt_stage[k][EXP_LO +: EXP_W];
        wire                     sgn_in = srt_stage[k][SGN_LO];
        wire [INST_FRM_BITS-1:0] frm_in = srt_stage[k][FRM_LO +: INST_FRM_BITS];
        wire [4:0]               exc_in = srt_stage[k][EXC_LO +: 5];
        wire                     nro_in = srt_stage[k][NRO_LO];
        wire                     sq_in  = srt_stage[k][SQ_LO];
        wire                     isd_in = srt_stage[k][ISD_LO];

        wire [W_BITS-1:0] ulp_a = nro_in ? ULP_A_NRO1 : ULP_A_NRO0;
        wire [W_BITS-1:0] ulp_b = nro_in ? ULP_B_NRO1 : ULP_B_NRO0;

        // --- Step A ---
        wire [W_BITS-1:0] D_S_a30 = sq_in ? (nro_in ? DS_in : {DS_in[W_BITS-2:0], 1'b0}) : DS_in;
        wire [W_BITS-1:0] D_ulp_a = sq_in ? (nro_in ? (ulp_a >> 1) : ulp_a) : '0;
        wire [W_BITS-1:0] val_a_add = D_S_a30 - D_ulp_a;
        wire [W_BITS-1:0] val_a_neg = D_S_a30 + D_ulp_a;
        wire [W_BITS-1:0] DS_a_plus  = DS_in + ulp_a;
        wire [W_BITS-1:0] DS_a_minus = DS_in - ulp_a;

        wire [CS_BITS:0] W_a_sum = {1'b0, Ws_in} + {1'b0, Wc_in};
        wire q_a = ~W_a_sum[CS_BITS-1];

        wire [CS_BITS-1:0] X_a    = q_a ? {1'b1, ~val_a_neg} : {1'b0, val_a_add};
        wire [CS_BITS-1:0] W2s_a  = {Ws_in[CS_BITS-2:0], q_a};
        wire [CS_BITS-1:0] W2c_a  = {Wc_in[CS_BITS-2:0], 1'b0};
        wire [CS_BITS-1:0] Ws_a   = W2s_a ^ W2c_a ^ X_a;
        wire [W_BITS-1:0]  Wca_raw = (W2s_a[W_BITS-1:0] & W2c_a[W_BITS-1:0]) | (W2c_a[W_BITS-1:0] & X_a[W_BITS-1:0]) | (W2s_a[W_BITS-1:0] & X_a[W_BITS-1:0]);
        wire [CS_BITS-1:0] Wc_a   = {Wca_raw, 1'b0};

        wire [W_BITS-1:0] DS_a = sq_in ? (q_a ? DS_a_plus : DS_a_minus) : DS_in;

        // --- Step B ---
        wire [W_BITS-1:0] D_S_b30 = sq_in ? (nro_in ? DS_a : {DS_a[W_BITS-2:0], 1'b0}) : DS_a;
        wire [W_BITS-1:0] D_ulp_b = sq_in ? (nro_in ? (ulp_b >> 1) : ulp_b) : '0;
        wire [W_BITS-1:0] val_b_add = D_S_b30 - D_ulp_b;
        wire [W_BITS-1:0] val_b_neg = D_S_b30 + D_ulp_b;
        wire [W_BITS-1:0] DS_b_plus  = DS_a + ulp_b;
        wire [W_BITS-1:0] DS_b_minus = DS_a - ulp_b;

        wire [CS_BITS:0] W_b_sum = {1'b0, Ws_a} + {1'b0, Wc_a};
        wire q_b = ~W_b_sum[CS_BITS-1];

        wire [CS_BITS-1:0] X_b    = q_b ? {1'b1, ~val_b_neg} : {1'b0, val_b_add};
        wire [CS_BITS-1:0] W2s_b  = {Ws_a[CS_BITS-2:0], q_b};
        wire [CS_BITS-1:0] W2c_b  = {Wc_a[CS_BITS-2:0], 1'b0};
        wire [CS_BITS-1:0] Ws_b   = W2s_b ^ W2c_b ^ X_b;
        wire [W_BITS-1:0]  Wcb_raw = (W2s_b[W_BITS-1:0] & W2c_b[W_BITS-1:0]) | (W2c_b[W_BITS-1:0] & X_b[W_BITS-1:0]) | (W2s_b[W_BITS-1:0] & X_b[W_BITS-1:0]);
        wire [CS_BITS-1:0] Wc_b   = {Wcb_raw, 1'b0};

        wire [W_BITS-1:0] DS_b = sq_in ? (q_b ? DS_b_plus : DS_b_minus) : DS_a;

        `UNUSED_VAR (qb_in[NR_BITS-1:NR_BITS-2])
        wire [NR_BITS-1:0] qb_new = {qb_in[NR_BITS-3:0], q_a, q_b};

        wire [STAGE_W-1:0] s_out;
        assign s_out[W_LO  +: CS_BITS]        = Ws_b;
        assign s_out[WC_LO +: CS_BITS]        = Wc_b;
        assign s_out[DS_LO +: W_BITS]         = DS_b;
        assign s_out[QB_LO +: NR_BITS]        = qb_new;
        assign s_out[QI_LO]                   = qi_in;
        assign s_out[EXP_LO +: EXP_W]         = exp_in;
        assign s_out[SGN_LO]                  = sgn_in;
        assign s_out[FRM_LO +: INST_FRM_BITS] = frm_in;
        assign s_out[EXC_LO +: 5]             = exc_in;
        assign s_out[NRO_LO]                  = nro_in;
        assign s_out[SQ_LO]                   = sq_in;
        assign s_out[ISD_LO]                  = isd_in;

        VX_pipe_register #(.DATAW (STAGE_W), .DEPTH (1)) srt_reg (
            .clk (clk), .reset (reset),
            .enable (enable && mask_pipe[PRE_LATENCY+INI_LATENCY-1+k]),
            .data_in (s_out), .data_out (srt_stage[k+1])
        );
    end

    // =========================================================================
    // CONV (combinational after srt_stage[NR_STAGES])
    // =========================================================================
    wire [CS_BITS-1:0]       Ws_cv  = srt_stage[NR_STAGES][W_LO  +: CS_BITS];
    wire [CS_BITS-1:0]       Wc_cv  = srt_stage[NR_STAGES][WC_LO +: CS_BITS];
    wire [W_BITS-1:0]        DS_cv  = srt_stage[NR_STAGES][DS_LO +: W_BITS];
    wire [NR_BITS-1:0]       qb_cv  = srt_stage[NR_STAGES][QB_LO +: NR_BITS];
    wire                     qi_cv  = srt_stage[NR_STAGES][QI_LO];
    wire [EXP_W-1:0]         exp_cv = srt_stage[NR_STAGES][EXP_LO +: EXP_W];
    wire                     sgn_cv = srt_stage[NR_STAGES][SGN_LO];
    wire [INST_FRM_BITS-1:0] frm_cv = srt_stage[NR_STAGES][FRM_LO +: INST_FRM_BITS];
    wire [4:0]               exc_cv = srt_stage[NR_STAGES][EXC_LO +: 5];
    wire                     nro_cv = srt_stage[NR_STAGES][NRO_LO];
    wire                     sq_cv  = srt_stage[NR_STAGES][SQ_LO];
    wire                     isd_cv = srt_stage[NR_STAGES][ISD_LO];

    wire signed [CS_BITS:0] W_cv_sum = $signed(Ws_cv) + $signed(Wc_cv);
    wire [W_BITS-1:0] W_cv = W_cv_sum[W_BITS-1:0];
    `UNUSED_VAR (W_cv_sum[CS_BITS:W_BITS])

    // ---- DIV CONV ----
    wire [NR_BITS:0] Q_frac = {qb_cv, 1'b0} - {1'b0, {NR_BITS{1'b1}}};
    `UNUSED_VAR (Q_frac[NR_BITS])
    wire [NR_BITS:0] Q_tot  = {qi_cv, Q_frac[NR_BITS-1:0]};

    wire              W_neg_div = W_cv[W_BITS-1];
    wire [NR_BITS:0]  Q_corr    = W_neg_div ? (Q_tot - 1'b1) : Q_tot;
    wire [W_BITS-1:0] W_corr_div = W_neg_div ? (W_cv + DS_cv) : W_cv;
    wire              sticky_div_r = (W_corr_div != '0);

    // Q_rnd: NR_BITS+1 bits, integer bit at MSB (NR_BITS) when qi=1.
    wire [NR_BITS:0] Q_rnd = qi_cv ? Q_corr : {Q_corr[NR_BITS-1:0], 1'b0};
    wire signed [EXP_W-1:0] exp_res_div = $signed(exp_cv) - (qi_cv ? '0 : EXP_W'(1));

    // Extract man/G/R/S from the top of Q_rnd at the active precision.
    // Built for both formats and muxed (avoids a variable part-select).
    wire [SUPER_SIG-1:0] man_div_d  = Q_rnd[NR_BITS -: SUPER_SIG];
    wire                 guard_div_d= Q_rnd[NR_BITS - SUPER_SIG];
    wire                 round_div_d= Q_rnd[NR_BITS - SUPER_SIG - 1];
    wire                 sticky_div_d=(|Q_rnd[NR_BITS - SUPER_SIG - 2 : 0]) | sticky_div_r;

    wire [23:0] man_div_s   = Q_rnd[NR_BITS -: 24];
    wire        guard_div_s = Q_rnd[NR_BITS - 24];
    wire        round_div_s = Q_rnd[NR_BITS - 25];
    wire        sticky_div_s = (|Q_rnd[NR_BITS - 26 : 0]) | sticky_div_r;

    // ---- SQRT CONV ----
    wire              W_neg_sq  = W_cv[W_BITS-1];
    wire [W_BITS-1:0] ulp_last  = nro_cv ? W_BITS'(1) : W_BITS'(2);
    wire [W_BITS-1:0] S_corr    = W_neg_sq ? (DS_cv - ulp_last) : DS_cv;

    wire [W_BITS:0]   W_cv_sx   = {W_cv[W_BITS-1], W_cv};
    wire [W_BITS:0]   W_D_sq    = nro_cv ? ({1'b0, DS_cv} - {1'b0, ulp_last})
                                         : ({DS_cv, 1'b0}  - {1'b0, ulp_last});
    wire [W_BITS:0]   W_corr_sq31 = W_neg_sq ? (W_cv_sx + W_D_sq) : W_cv_sx;
    `UNUSED_VAR (W_corr_sq31[W_BITS])
    wire sticky_sq_w  = (W_corr_sq31[W_BITS-1:0] != '0);

    // S_corr leading bit at SQRT_LEAD; man=[LEAD -: act_sig], guard=[LEAD-act_sig], etc.
    wire [SUPER_SIG-1:0] man_sq_d   = S_corr[SQRT_LEAD -: SUPER_SIG];
    wire                 guard_sq_d = S_corr[SQRT_LEAD - SUPER_SIG];
    wire                 round_sq_d = S_corr[SQRT_LEAD - SUPER_SIG - 1];
    wire                 sticky_sq_d= (|S_corr[SQRT_LEAD - SUPER_SIG - 2 : 0]) | sticky_sq_w;

    wire [23:0] man_sq_s   = S_corr[SQRT_LEAD -: 24];
    wire        guard_sq_s = S_corr[SQRT_LEAD - 24];
    wire        round_sq_s = S_corr[SQRT_LEAD - 25];
    wire        sticky_sq_s = (|S_corr[SQRT_LEAD - 26 : 0]) | sticky_sq_w;
    `UNUSED_VAR (S_corr[W_BITS-1:SQRT_LEAD+1])

    wire signed [EXP_W-1:0] exp_res_sq = $signed(exp_cv);

    // ---- Unified CONV output (mux format then op) ----
    wire act_s = (HAS_D && !isd_cv);   // active F32 extraction
    wire [SUPER_SIG-1:0] man_div = act_s ? SUPER_SIG'(man_div_s) : man_div_d;
    wire [SUPER_SIG-1:0] man_sq  = act_s ? SUPER_SIG'(man_sq_s)  : man_sq_d;
    wire guard_div  = act_s ? guard_div_s  : guard_div_d;
    wire round_div  = act_s ? round_div_s  : round_div_d;
    wire sticky_div = act_s ? sticky_div_s : sticky_div_d;
    wire guard_sq   = act_s ? guard_sq_s   : guard_sq_d;
    wire round_sq   = act_s ? round_sq_s   : round_sq_d;
    wire sticky_sq  = act_s ? sticky_sq_s  : sticky_sq_d;

    wire [SUPER_SIG-1:0] man_cv_out   = sq_cv ? man_sq    : man_div;
    wire                 guard_cv_out = sq_cv ? guard_sq  : guard_div;
    wire                 round_cv_out = sq_cv ? round_sq  : round_div;
    wire                 sticky_cv_out= sq_cv ? sticky_sq : sticky_div;
    wire signed [EXP_W-1:0] exp_res   = sq_cv ? exp_res_sq : exp_res_div;

    // =========================================================================
    // CONV pipeline register
    // =========================================================================
    localparam CONV_W = SUPER_SIG + 1 + 1 + 1 + EXP_W + 1 + INST_FRM_BITS + 5 + 1;

    wire [CONV_W-1:0] conv_in = {man_cv_out, guard_cv_out, round_cv_out, sticky_cv_out,
                                  exp_res, sgn_cv, frm_cv, exc_cv, isd_cv};
    wire [CONV_W-1:0] conv_out;
    VX_pipe_register #(.DATAW (CONV_W), .DEPTH (1)) conv_reg (
        .clk (clk), .reset (reset), .enable (enable && valid_conv),
        .data_in (conv_in), .data_out (conv_out)
    );

    // =========================================================================
    // NRM stage: round, overflow/underflow, pack at active format.
    // =========================================================================
    wire [SUPER_SIG-1:0]     s_man;
    wire                     s_guard, s_round, s_sticky;
    wire [EXP_W-1:0]         s_exp_bits;
    wire                     s_sign;
    wire [INST_FRM_BITS-1:0] s_frm;
    wire [4:0]               s_exc;
    wire                     s_isd;

    assign {s_man, s_guard, s_round, s_sticky, s_exp_bits, s_sign, s_frm, s_exc, s_isd} = conv_out;
    wire signed [EXP_W-1:0] s_exp = $signed(s_exp_bits);

    wire act_d = HAS_D ? s_isd : 1'b0;   // double result

    // Round at the active mantissa width.  man is SUPER_SIG bits with the
    // integer bit at SUPER_SIG-1; F32 result man occupies the top 24 bits.
    wire [SUPER_SIG-1:0] abs_rounded;
    wire        round_sign, exact_zero;
    VX_fp_rounding #(.DAT_WIDTH(SUPER_SIG)) u_rnd (
        .abs_value_i            (s_man),
        .sign_i                 (s_sign),
        .round_sticky_bits_i    ({s_guard, s_round | s_sticky}),
        .rnd_mode_i             (s_frm),
        .effective_subtraction_i(1'b0),
        .abs_rounded_o          (abs_rounded),
        .sign_o                 (round_sign),
        .exact_zero_o           (exact_zero)
    );

    // mantissa is right-justified at the active width (integer bit at SUPER_SIG-1
    // for F64, at bit 23 for F32). Rounding-carry overflows the active integer bit.
    wire round_carry;
    if (HAS_D) begin : g_rcarry_d
        wire round_carry_d = (abs_rounded == '0) & (s_man != '0); // 53-bit wrap (F64)
        wire round_carry_s = abs_rounded[24];                     // carry out of 24-bit man (F32)
        assign round_carry = act_d ? round_carry_d : round_carry_s;
    end else begin : g_rcarry_s
        assign round_carry = (abs_rounded == '0) & (s_man != '0); // 24-bit wrap
    end
    wire signed [EXP_W-1:0] fin_exp = s_exp + (round_carry ? EXP_W'(1) : '0);

    // active mantissa field
    wire [SUPER_MAN-1:0] fin_man_d = round_carry ? '0 : abs_rounded[SUPER_MAN-1:0];
    wire [22:0]          fin_man_s = round_carry ? 23'd0 : abs_rounded[22:0];

    wire is_nan  = s_exc[4];
    wire is_inf  = s_exc[3];
    wire is_zero = s_exc[2];
    wire dz_flag = s_exc[1];
    wire nv_flag = s_exc[0];

    // active exponent all-ones / range
    wire [EXP_W-1:0] act_allones = act_d ? EXP_W'(2047) : EXP_W'(255);
    wire of_flag = (fin_exp >= $signed(act_allones)) & ~is_nan & ~is_inf;
    wire uf_flag = (fin_exp <= '0) & ~is_nan & ~is_inf & ~is_zero & ~exact_zero;
    wire nx_flag = (s_guard | s_round | s_sticky) & ~is_nan & ~is_inf;

    // F32 pack (always present)
    wire [31:0] nan_s  = 32'h7FC00000;
    wire [31:0] inf_s  = {round_sign, 8'hFF, 23'd0};
    wire [31:0] zero_s = {round_sign, 31'd0};
    wire [31:0] norm_s = {round_sign, fin_exp[7:0], fin_man_s};

    reg [31:0] res_s;
    always @(*) begin
        if (is_nan)                              res_s = nan_s;
        else if (is_inf | of_flag)               res_s = inf_s;
        else if (is_zero | exact_zero | uf_flag) res_s = zero_s;
        else                                     res_s = norm_s;
    end

    // Pack into FLEN: F64 fills the width; F32 is NaN-boxed (upper bits ones).
    // The F64 packing is only elaborated when HAS_D (fin_exp/fin_man are wide enough).
    wire [FLEN-1:0] nrm_result;
    if (HAS_D) begin : g_pack_d
        wire [63:0] nan_d  = 64'h7FF8000000000000;
        wire [63:0] inf_d  = {round_sign, 11'h7FF, 52'd0};
        wire [63:0] zero_d = {round_sign, 63'd0};
        wire [63:0] norm_d = {round_sign, fin_exp[10:0], fin_man_d};
        reg [63:0] res_d;
        always @(*) begin
            if (is_nan)                              res_d = nan_d;
            else if (is_inf | of_flag)               res_d = inf_d;
            else if (is_zero | exact_zero | uf_flag) res_d = zero_d;
            else                                     res_d = norm_d;
        end
        assign nrm_result = act_d ? FLEN'(res_d) : {{(FLEN-32){1'b1}}, res_s};
    end else begin : g_pack_s
        assign nrm_result = FLEN'(res_s);
        `UNUSED_VAR (fin_man_d)
    end

    fflags_t nrm_fflags;
    assign nrm_fflags.NV = nv_flag;
    assign nrm_fflags.DZ = dz_flag;
    assign nrm_fflags.OF = of_flag;
    assign nrm_fflags.UF = uf_flag;
    assign nrm_fflags.NX = nx_flag | of_flag | uf_flag;

    VX_pipe_register #(.DATAW (FLEN + `FP_FLAGS_BITS), .DEPTH (1)) nrm_reg (
        .clk (clk), .reset (reset), .enable (enable && valid_nrm),
        .data_in  ({nrm_result, nrm_fflags}),
        .data_out ({result,     fflags})
    );

endmodule
