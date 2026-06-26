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

// Single-lane pipelined FDIV (pure RTL, single-function).
//
// Specialization of VX_fdivsqrt_unit with the SQRT path removed: the same
// non-restoring radix-2 carry-save NR recurrence, sized for the widest
// supported format (SUPER = F64 when FLEN>=64, else F32). Only the front
// (unpack) and back (extract/round/pack) stages are format-aware:
//   - operands are unpacked at their active format (F32/F64) and the
//     normalized significand is LEFT-justified into the SUPER significand,
//   - the exponent is tracked in the ACTIVE format's frame (bias/range), so
//     an F32 op rounds ONCE at 24-bit precision (no double rounding) and
//     under/overflows at the F32 boundary.
//
// Pipeline: 1 PRE + 1 INI + NR_STAGES SRT + 1 CONV + 1 NRM.
//   SUPER=F32: NR_STAGES=13 -> LATENCY 17
//   SUPER=F64: NR_STAGES=28 -> LATENCY 32
//
// DIV: Non-restoring radix-2 NR.  W, D scaled x2^SCALE_DIV; NR invariant |W|<=D.

`include "VX_fpu_define.vh"

module VX_fdiv_unit_rtl import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY = 17,
    parameter FLEN    = 32,
    // Reserved for a future vendor/DSP backend select. The SRT datapath has no
    // multiplier, so it is a no-op here; accepted for a uniform FPU unit API.
    parameter USE_DSP = 0,
    // 1: full IEEE subnormal support. 0: flush-to-zero (DAZ subnormal inputs to
    //    signed zero + FTZ subnormal results) for area — drops the input
    //    normalization. Use 0 for relaxed-precision datapaths (e.g. RTU).
    parameter SNORM_ENABLE = 1,
    // 1: detect NaN/inf/div-by-zero and produce IEEE special results + fflags.
    // 0: assume finite, non-exceptional operands; drop the exception cone and
    //    tie fflags to 0 (area). Use 0 for the RTU geometry path.
    parameter EXCEPT_ENABLE = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire mask,

    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [FLEN-1:0] dataa,   // dividend
    input  wire [FLEN-1:0] datab,   // divisor

    output wire [FLEN-1:0] result,
    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam HAS_D     = (FLEN >= 64);
    wire is_d = HAS_D ? fmt[0] : 1'b0;
    `UNUSED_VAR (fmt)
    `UNUSED_PARAM (USE_DSP)

    // ---- SUPER (widest) format the datapath is built for ----
    localparam SUPER_MAN = HAS_D ? 52 : 23;
    localparam SUPER_SIG = SUPER_MAN + 1;                 // 24 / 53

    localparam PRE_LATENCY  = 1;
    localparam INI_LATENCY  = 1;
    localparam CONV_LATENCY = 1;
    localparam NRM_LATENCY  = 1;

    localparam SCALE_DIV  = 5;

    localparam W_BITS   = SUPER_SIG + SCALE_DIV + 1;      // 30 / 59
    localparam CS_BITS  = W_BITS + 1;                     // 31 / 60: carry-save component width
    localparam NR_BITS  = (SUPER_SIG + 2) + ((SUPER_SIG + 2) % 2); // even: 26 / 56
    localparam NR_STAGES = NR_BITS / 2;                  // 13 / 28
    localparam EXP_W    = HAS_D ? 14 : 10;               // signed exponent working width

    `STATIC_ASSERT(LATENCY == (PRE_LATENCY + INI_LATENCY + NR_STAGES + CONV_LATENCY + NRM_LATENCY), ("VX_fdiv_unit: LATENCY must be %0d, got %0d", PRE_LATENCY+INI_LATENCY+NR_STAGES+CONV_LATENCY+NRM_LATENCY, LATENCY))

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
    localparam DS_LO   = QB_LO   + NR_BITS;               // D (divisor)
    localparam W_LO    = DS_LO   + W_BITS;                // Ws (CS_BITS wide)
    localparam WC_LO   = W_LO    + CS_BITS;               // Wc (CS_BITS wide)
    localparam ISD_LO  = WC_LO   + CS_BITS;               // is_double
    localparam STAGE_W = ISD_LO  + 1;

    // =========================================================================
    // Active-format unpack: classify, LZC-normalize the active significand, then
    // LEFT-justify it into the SUPER significand. Exponent stays in active frame.
    // =========================================================================
    wire [EXP_W-1:0] BIAS = is_d ? EXP_W'(1023) : EXP_W'(127);

    // DAZ: when subnormals are disabled, flush subnormal inputs to signed zero
    // before unpacking. The input LZC normalizers then go dead and are pruned.
    localparam DAZ = (SNORM_ENABLE == 0);
    wire [FLEN-1:0] da, db;
    if (HAS_D) begin : g_daz
        wire as = is_d ? ((dataa[62:52]=='0) && (|dataa[51:0])) : ((dataa[30:23]=='0) && (|dataa[22:0]));
        wire bs = is_d ? ((datab[62:52]=='0) && (|datab[51:0])) : ((datab[30:23]=='0) && (|datab[22:0]));
        assign da = (DAZ & as) ? {dataa[FLEN-1], {(FLEN-1){1'b0}}} : dataa;
        assign db = (DAZ & bs) ? {datab[FLEN-1], {(FLEN-1){1'b0}}} : datab;
    end else begin : g_daz_s
        wire as = (dataa[30:23]=='0) && (|dataa[22:0]);
        wire bs = (datab[30:23]=='0) && (|datab[22:0]);
        assign da = (DAZ & as) ? {dataa[31], 31'b0} : dataa;
        assign db = (DAZ & bs) ? {datab[31], 31'b0} : datab;
    end

    // -- raw active field extraction --
    wire s_a = is_d ? da[FLEN-1] : da[31];
    wire s_b = is_d ? db[FLEN-1] : db[31];

    // active exponent / mantissa selectors and classification
    wire ea_allones, eb_allones, ea_zero, eb_zero, ma_nz, mb_nz, ma_q, mb_q;
    wire [SUPER_SIG-1:0] siga_ljn, sigb_ljn;   // normalized, left-justified into SUPER_SIG
    wire signed [EXP_W-1:0] exp_a, exp_b;

    // --- F32 view ---
    wire        f32a_z  = (da[30:23] == 8'd0) && (da[22:0] == 23'd0);
    wire        f32a_sub= (da[30:23] == 8'd0) && (da[22:0] != 23'd0);
    wire        f32b_z  = (db[30:23] == 8'd0) && (db[22:0] == 23'd0);
    wire        f32b_sub= (db[30:23] == 8'd0) && (db[22:0] != 23'd0);
    wire [23:0] f32a_sig = {~(f32a_z | f32a_sub), da[22:0]};
    wire [23:0] f32b_sig = {~(f32b_z | f32b_sub), db[22:0]};
    wire [4:0]  f32a_lzc, f32b_lzc;
    wire f32a_lvld, f32b_lvld;
    VX_lzc #(.N(24)) lz_f32a (.data_in(f32a_sig), .data_out(f32a_lzc), .valid_out(f32a_lvld));
    VX_lzc #(.N(24)) lz_f32b (.data_in(f32b_sig), .data_out(f32b_lzc), .valid_out(f32b_lvld));
    wire [23:0] f32a_norm = f32a_sub ? (f32a_sig << f32a_lzc) : f32a_sig;
    wire [23:0] f32b_norm = f32b_sub ? (f32b_sig << f32b_lzc) : f32b_sig;
    wire signed [EXP_W-1:0] f32a_exp = f32a_sub ? (EXP_W'(1) - EXP_W'(f32a_lzc)) : f32a_z ? '0 : EXP_W'(da[30:23]);
    wire signed [EXP_W-1:0] f32b_exp = f32b_sub ? (EXP_W'(1) - EXP_W'(f32b_lzc)) : f32b_z ? '0 : EXP_W'(db[30:23]);
    `UNUSED_VAR ({f32a_lvld, f32b_lvld})

    if (HAS_D) begin : g_unpack_d
        wire        f64a_z  = (da[62:52] == 11'd0) && (da[51:0] == 52'd0);
        wire        f64a_sub= (da[62:52] == 11'd0) && (da[51:0] != 52'd0);
        wire        f64b_z  = (db[62:52] == 11'd0) && (db[51:0] == 52'd0);
        wire        f64b_sub= (db[62:52] == 11'd0) && (db[51:0] != 52'd0);
        wire [52:0] f64a_sig = {~(f64a_z | f64a_sub), da[51:0]};
        wire [52:0] f64b_sig = {~(f64b_z | f64b_sub), db[51:0]};
        wire [5:0]  f64a_lzc, f64b_lzc;
        wire f64a_lvld, f64b_lvld;
        VX_lzc #(.N(53)) lz_f64a (.data_in(f64a_sig), .data_out(f64a_lzc), .valid_out(f64a_lvld));
        VX_lzc #(.N(53)) lz_f64b (.data_in(f64b_sig), .data_out(f64b_lzc), .valid_out(f64b_lvld));
        wire [52:0] f64a_norm = f64a_sub ? (f64a_sig << f64a_lzc) : f64a_sig;
        wire [52:0] f64b_norm = f64b_sub ? (f64b_sig << f64b_lzc) : f64b_sig;
        wire signed [EXP_W-1:0] f64a_exp = f64a_sub ? (EXP_W'(1) - EXP_W'(f64a_lzc)) : f64a_z ? '0 : EXP_W'(da[62:52]);
        wire signed [EXP_W-1:0] f64b_exp = f64b_sub ? (EXP_W'(1) - EXP_W'(f64b_lzc)) : f64b_z ? '0 : EXP_W'(db[62:52]);
        `UNUSED_VAR ({f64a_lvld, f64b_lvld})

        assign ea_allones = is_d ? (&da[62:52]) : (&da[30:23]);
        assign eb_allones = is_d ? (&db[62:52]) : (&db[30:23]);
        assign ea_zero    = is_d ? f64a_z  : f32a_z;
        assign eb_zero    = is_d ? f64b_z  : f32b_z;
        assign ma_nz      = is_d ? (|da[51:0]) : (|da[22:0]);
        assign mb_nz      = is_d ? (|db[51:0]) : (|db[22:0]);
        assign ma_q       = is_d ? da[51] : da[22];   // quiet bit (msb of man)
        assign mb_q       = is_d ? db[51] : db[22];
        assign siga_ljn   = is_d ? f64a_norm : {f32a_norm, {(SUPER_SIG-24){1'b0}}};
        assign sigb_ljn   = is_d ? f64b_norm : {f32b_norm, {(SUPER_SIG-24){1'b0}}};
        assign exp_a      = is_d ? f64a_exp : f32a_exp;
        assign exp_b      = is_d ? f64b_exp : f32b_exp;
    end else begin : g_unpack_s
        assign ea_allones = &da[30:23];
        assign eb_allones = &db[30:23];
        assign ea_zero    = f32a_z;
        assign eb_zero    = f32b_z;
        assign ma_nz      = |da[22:0];
        assign mb_nz      = |db[22:0];
        assign ma_q       = da[22];
        assign mb_q       = db[22];
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
    wire dz0_div  = zero_b & ~nan_a & ~nan_b & ~zero_a & ~inf_a; // DZ only for finite x/0
    wire rnan_div = nan_a | nan_b | nv0_div;
    wire rinf_div = (inf_a | zero_b) & ~rnan_div;
    wire rzro_div = (zero_a | inf_b) & ~rnan_div;
    // EXCEPT_ENABLE=0 assumes finite, non-exceptional operands: drop the NaN/
    // inf/DZ cone (exc tied off -> NRM takes the normal pack) and fflags.
    wire [4:0] exc0_div = EXCEPT_ENABLE ? {rnan_div, rinf_div, rzro_div, dz0_div, nv0_div} : 5'b0;

    // =========================================================================
    // PRE pipeline register (cycle 1): normalize + exc.
    // =========================================================================
    localparam INI_SIG_LO   = 0;
    localparam INI_SIGB_LO  = INI_SIG_LO   + SUPER_SIG;
    localparam INI_EXPD_LO  = INI_SIGB_LO  + SUPER_SIG;
    localparam INI_SGND_LO  = INI_EXPD_LO  + EXP_W;
    localparam INI_EXCDV_LO = INI_SGND_LO  + 1;
    localparam INI_FRM_LO   = INI_EXCDV_LO + 5;
    localparam INI_ISD_LO   = INI_FRM_LO   + INST_FRM_BITS;
    localparam INI_W        = INI_ISD_LO   + 1;

    wire [INI_W-1:0] ini_in;
    assign ini_in[INI_SIG_LO   +: SUPER_SIG]     = siga_ljn;
    assign ini_in[INI_SIGB_LO  +: SUPER_SIG]     = sigb_ljn;
    assign ini_in[INI_EXPD_LO  +: EXP_W]         = exp_r0_div;
    assign ini_in[INI_SGND_LO]                   = sign_r0_div;
    assign ini_in[INI_EXCDV_LO +: 5]             = exc0_div;
    assign ini_in[INI_FRM_LO   +: INST_FRM_BITS] = frm;
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
    wire                     i_sgn_div = ini_out[INI_SGND_LO];
    wire [4:0]               i_exc_div = ini_out[INI_EXCDV_LO +: 5];
    wire [INST_FRM_BITS-1:0] i_frm     = ini_out[INI_FRM_LO   +: INST_FRM_BITS];
    wire                     i_isd     = ini_out[INI_ISD_LO];

    wire [W_BITS-1:0]    i_D0_div     = {1'b0, i_sig_b, {SCALE_DIV{1'b0}}};
    wire                 i_q_int0_div = (i_sig_a >= i_sig_b);
    wire [SUPER_SIG-1:0] i_pre_diff   = i_sig_a - (i_q_int0_div ? i_sig_b : {SUPER_SIG{1'b0}});
    wire [W_BITS-1:0]    i_W0_div     = {1'b0, i_pre_diff, {SCALE_DIV{1'b0}}};

    wire [STAGE_W-1:0] srt_stage [0:NR_STAGES];
    wire [STAGE_W-1:0] pre_in;

    assign pre_in[W_LO  +: CS_BITS]        = {1'b0, i_W0_div};
    assign pre_in[WC_LO +: CS_BITS]        = '0;
    assign pre_in[DS_LO +: W_BITS]         = i_D0_div;
    assign pre_in[QB_LO +: NR_BITS]        = {NR_BITS{1'b0}};
    assign pre_in[QI_LO]                   = i_q_int0_div;
    assign pre_in[EXP_LO +: EXP_W]         = i_exp_div;
    assign pre_in[SGN_LO]                  = i_sgn_div;
    assign pre_in[FRM_LO +: INST_FRM_BITS] = i_frm;
    assign pre_in[EXC_LO +: 5]             = i_exc_div;
    assign pre_in[ISD_LO]                  = i_isd;

    VX_pipe_register #(.DATAW (STAGE_W), .DEPTH (1)) ini_reg (
        .clk (clk), .reset (reset), .enable (enable && valid_ini),
        .data_in (pre_in), .data_out (srt_stage[0])
    );

    // =========================================================================
    // SRT stages 1..NR_STAGES (two NR steps each, carry-save W). The divisor D
    // is constant through the recurrence; X = q ? {1,~D} : {0,D}.
    // =========================================================================
    for (genvar k = 0; k < NR_STAGES; k++) begin : g_srt
        wire [CS_BITS-1:0]       Ws_in  = srt_stage[k][W_LO  +: CS_BITS];
        wire [CS_BITS-1:0]       Wc_in  = srt_stage[k][WC_LO +: CS_BITS];
        wire [W_BITS-1:0]        DS_in  = srt_stage[k][DS_LO +: W_BITS];
        wire [NR_BITS-1:0]       qb_in  = srt_stage[k][QB_LO +: NR_BITS];
        wire                     qi_in  = srt_stage[k][QI_LO];
        wire [EXP_W-1:0]         exp_in = srt_stage[k][EXP_LO +: EXP_W];
        wire                     sgn_in = srt_stage[k][SGN_LO];
        wire [INST_FRM_BITS-1:0] frm_in = srt_stage[k][FRM_LO +: INST_FRM_BITS];
        wire [4:0]               exc_in = srt_stage[k][EXC_LO +: 5];
        wire                     isd_in = srt_stage[k][ISD_LO];

        // --- Step A ---
        wire [CS_BITS:0] W_a_sum = {1'b0, Ws_in} + {1'b0, Wc_in};
        wire q_a = ~W_a_sum[CS_BITS-1];

        wire [CS_BITS-1:0] X_a    = q_a ? {1'b1, ~DS_in} : {1'b0, DS_in};
        wire [CS_BITS-1:0] W2s_a  = {Ws_in[CS_BITS-2:0], q_a};
        wire [CS_BITS-1:0] W2c_a  = {Wc_in[CS_BITS-2:0], 1'b0};
        wire [CS_BITS-1:0] Ws_a   = W2s_a ^ W2c_a ^ X_a;
        wire [W_BITS-1:0]  Wca_raw = (W2s_a[W_BITS-1:0] & W2c_a[W_BITS-1:0]) | (W2c_a[W_BITS-1:0] & X_a[W_BITS-1:0]) | (W2s_a[W_BITS-1:0] & X_a[W_BITS-1:0]);
        wire [CS_BITS-1:0] Wc_a   = {Wca_raw, 1'b0};

        // --- Step B ---
        wire [CS_BITS:0] W_b_sum = {1'b0, Ws_a} + {1'b0, Wc_a};
        wire q_b = ~W_b_sum[CS_BITS-1];

        wire [CS_BITS-1:0] X_b    = q_b ? {1'b1, ~DS_in} : {1'b0, DS_in};
        wire [CS_BITS-1:0] W2s_b  = {Ws_a[CS_BITS-2:0], q_b};
        wire [CS_BITS-1:0] W2c_b  = {Wc_a[CS_BITS-2:0], 1'b0};
        wire [CS_BITS-1:0] Ws_b   = W2s_b ^ W2c_b ^ X_b;
        wire [W_BITS-1:0]  Wcb_raw = (W2s_b[W_BITS-1:0] & W2c_b[W_BITS-1:0]) | (W2c_b[W_BITS-1:0] & X_b[W_BITS-1:0]) | (W2s_b[W_BITS-1:0] & X_b[W_BITS-1:0]);
        wire [CS_BITS-1:0] Wc_b   = {Wcb_raw, 1'b0};

        `UNUSED_VAR (qb_in[NR_BITS-1:NR_BITS-2])
        wire [NR_BITS-1:0] qb_new = {qb_in[NR_BITS-3:0], q_a, q_b};

        wire [STAGE_W-1:0] s_out;
        assign s_out[W_LO  +: CS_BITS]        = Ws_b;
        assign s_out[WC_LO +: CS_BITS]        = Wc_b;
        assign s_out[DS_LO +: W_BITS]         = DS_in;
        assign s_out[QB_LO +: NR_BITS]        = qb_new;
        assign s_out[QI_LO]                   = qi_in;
        assign s_out[EXP_LO +: EXP_W]         = exp_in;
        assign s_out[SGN_LO]                  = sgn_in;
        assign s_out[FRM_LO +: INST_FRM_BITS] = frm_in;
        assign s_out[EXC_LO +: 5]             = exc_in;
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
    wire [SUPER_SIG-1:0] man_div_d  = Q_rnd[NR_BITS -: SUPER_SIG];
    wire                 guard_div_d= Q_rnd[NR_BITS - SUPER_SIG];
    wire                 round_div_d= Q_rnd[NR_BITS - SUPER_SIG - 1];
    wire                 sticky_div_d=(|Q_rnd[NR_BITS - SUPER_SIG - 2 : 0]) | sticky_div_r;

    wire [23:0] man_div_s   = Q_rnd[NR_BITS -: 24];
    wire        guard_div_s = Q_rnd[NR_BITS - 24];
    wire        round_div_s = Q_rnd[NR_BITS - 25];
    wire        sticky_div_s = (|Q_rnd[NR_BITS - 26 : 0]) | sticky_div_r;

    // ---- Unified CONV output (mux format) ----
    wire act_s = (HAS_D && !isd_cv);   // active F32 extraction
    wire [SUPER_SIG-1:0] man_cv_out   = act_s ? SUPER_SIG'(man_div_s) : man_div_d;
    wire                 guard_cv_out = act_s ? guard_div_s  : guard_div_d;
    wire                 round_cv_out = act_s ? round_div_s  : round_div_d;
    wire                 sticky_cv_out= act_s ? sticky_div_s : sticky_div_d;
    wire signed [EXP_W-1:0] exp_res   = exp_res_div;

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

    wire is_nan  = s_exc[4];
    wire is_inf  = s_exc[3];
    wire is_zero = s_exc[2];
    wire dz_flag = s_exc[1];
    wire nv_flag = s_exc[0];

    // Subnormal handling. A biased result exponent <= 0 is subnormal/underflow.
    // With SNORM_ENABLE the significand is denormalized (right-shifted by 1-s_exp)
    // so it rounds ONCE at the subnormal LSB and a true subnormal is emitted;
    // shifted-out bits fold into guard/sticky. Without it (FTZ), dsh=0 and the
    // result is flushed to signed zero in the pack below.
    localparam SH_W = `CLOG2(SUPER_SIG + 2) + 1;
    wire result_sub = ($signed(s_exp) <= 0) & ~is_nan & ~is_inf & ~is_zero;
    wire is_sub_res = (SNORM_ENABLE != 0) & result_sub;
    wire signed [EXP_W-1:0] denorm_amt = is_sub_res ? (EXP_W'(1) - s_exp) : '0;
    wire huge_denorm = is_sub_res & ($signed(denorm_amt) >= $signed(EXP_W'(SUPER_SIG + 1)));
    wire [SH_W-1:0] dsh = huge_denorm ? SH_W'(SUPER_SIG) : SH_W'(denorm_amt);

    wire [SUPER_SIG-1:0] below_gmsk = (dsh <= 1) ? '0 : ((SUPER_SIG'(1) << (dsh - 1)) - SUPER_SIG'(1));
    wire [SUPER_SIG-1:0] sub_man = s_man >> dsh;
    wire sub_guard  = (dsh == 0)  ? s_guard
                    : huge_denorm ? 1'b0
                                  : s_man[dsh - 1];
    wire sub_sticky = (dsh == 0) ? (s_round | s_sticky)
                    : ((huge_denorm ? (|s_man) : (|(s_man & below_gmsk))) | s_guard | s_round | s_sticky);

    // Round at the (possibly denormalized) active mantissa width.
    wire [SUPER_SIG-1:0] abs_rounded;
    wire        round_sign, exact_zero;
    VX_fp_rounding #(.DAT_WIDTH(SUPER_SIG)) u_rnd (
        .abs_value_i            (sub_man),
        .sign_i                 (s_sign),
        .round_sticky_bits_i    ({sub_guard, sub_sticky}),
        .rnd_mode_i             (s_frm),
        .effective_subtraction_i(1'b0),
        .abs_rounded_o          (abs_rounded),
        .sign_o                 (round_sign),
        .exact_zero_o           (exact_zero)
    );

    // Carry: normal -> exponent++ on integer-bit carry; subnormal that rounds up
    // sets the hidden bit -> smallest normal (biased exp 1).
    wire sub_carry = abs_rounded[SUPER_MAN];
    wire norm_carry;
    if (HAS_D) begin : g_rcarry_d
        wire rc_d = (abs_rounded == '0) & (sub_man != '0); // 53-bit wrap (F64)
        wire rc_s = abs_rounded[24];                       // carry out of 24-bit man (F32)
        assign norm_carry = act_d ? rc_d : rc_s;
    end else begin : g_rcarry_s
        assign norm_carry = (abs_rounded == '0) & (sub_man != '0); // 24-bit wrap
    end

    wire signed [EXP_W-1:0] fin_exp = is_sub_res ? (sub_carry ? EXP_W'(1) : '0)
                                                 : (s_exp + (norm_carry ? EXP_W'(1) : '0));

    // active mantissa field (subnormal mantissa falls out of abs_rounded low bits)
    wire [SUPER_MAN-1:0] fin_man_d = abs_rounded[SUPER_MAN-1:0];
    wire [22:0]          fin_man_s = abs_rounded[22:0];

    // active exponent all-ones / range
    wire [EXP_W-1:0] act_allones = act_d ? EXP_W'(2047) : EXP_W'(255);
    wire of_flag = ($signed(fin_exp) >= $signed(act_allones)) & ~is_nan & ~is_inf;
    wire nx_flag = (sub_guard | sub_sticky) & ~is_nan & ~is_inf & ~is_zero;
    wire uf_flag = result_sub & nx_flag;          // tiny + inexact
    wire ftz_flush = result_sub & (SNORM_ENABLE == 0); // FTZ: flush underflow to 0

    // Overflow result is max-normal vs infinity per rounding mode/sign (IEEE):
    // RTZ always max-normal; RDN max-normal if positive; RUP max-normal if negative.
    wire ovf_to_max = (s_frm == INST_FRM_RTZ)
                    | (s_frm == INST_FRM_RDN & ~round_sign)
                    | (s_frm == INST_FRM_RUP &  round_sign);

    // F32 pack (always present)
    wire [31:0] nan_s     = 32'h7FC00000;
    wire [31:0] inf_s     = {round_sign, 8'hFF, 23'd0};
    wire [31:0] maxnorm_s = {round_sign, 8'hFE, 23'h7FFFFF};
    wire [31:0] zero_s    = {round_sign, 31'd0};
    wire [31:0] norm_s    = {round_sign, fin_exp[7:0], fin_man_s};

    reg [31:0] res_s;
    always @(*) begin
        if (is_nan)                              res_s = nan_s;
        else if (is_inf)                         res_s = inf_s;
        else if (of_flag)                        res_s = ovf_to_max ? maxnorm_s : inf_s;
        else if (is_zero | exact_zero | ftz_flush) res_s = zero_s;
        else                                     res_s = norm_s;
    end

    // Pack into FLEN: F64 fills the width; F32 is NaN-boxed (upper bits ones).
    wire [FLEN-1:0] nrm_result;
    if (HAS_D) begin : g_pack_d
        wire [63:0] nan_d     = 64'h7FF8000000000000;
        wire [63:0] inf_d     = {round_sign, 11'h7FF, 52'd0};
        wire [63:0] maxnorm_d = {round_sign, 11'h7FE, 52'hFFFFFFFFFFFFF};
        wire [63:0] zero_d    = {round_sign, 63'd0};
        wire [63:0] norm_d    = {round_sign, fin_exp[10:0], fin_man_d};
        reg [63:0] res_d;
        always @(*) begin
            if (is_nan)                              res_d = nan_d;
            else if (is_inf)                         res_d = inf_d;
            else if (of_flag)                        res_d = ovf_to_max ? maxnorm_d : inf_d;
            else if (is_zero | exact_zero | ftz_flush) res_d = zero_d;
            else                                     res_d = norm_d;
        end
        assign nrm_result = act_d ? FLEN'(res_d) : {{(FLEN-32){1'b1}}, res_s};
    end else begin : g_pack_s
        assign nrm_result = FLEN'(res_s);
        `UNUSED_VAR (fin_man_d)
    end

    fflags_t nrm_fflags;
    assign nrm_fflags.NV = EXCEPT_ENABLE & nv_flag;
    assign nrm_fflags.DZ = EXCEPT_ENABLE & dz_flag;
    assign nrm_fflags.OF = EXCEPT_ENABLE & of_flag;
    assign nrm_fflags.UF = EXCEPT_ENABLE & uf_flag;
    assign nrm_fflags.NX = EXCEPT_ENABLE & (nx_flag | of_flag | uf_flag);

    VX_pipe_register #(.DATAW (FLEN + `FP_FLAGS_BITS), .DEPTH (1)) nrm_reg (
        .clk (clk), .reset (reset), .enable (enable && valid_nrm),
        .data_in  ({nrm_result, nrm_fflags}),
        .data_out ({result,     fflags})
    );

endmodule
