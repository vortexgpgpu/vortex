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

// Modified port of noncomp module from fpnew Libray
// reference: https://github.com/pulp-platform/fpnew

// Merged sign-inject / min-max / compare / classify / move unit.
// FLEN=32 handles F32 only; FLEN=64 adds F64, selected per-op by fmt[0]
// (0=single, 1=double). F32 operands occupy the low 32 bits (NaN-boxed).

`include "VX_fpu_define.vh"

module VX_fncp_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 1,
    parameter FLEN     = 32,
    parameter OUT_REG  = 0
) (
    input wire clk,
    input wire reset,

    input wire enable,
    input wire mask,

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] fmt,
    input wire [INST_FRM_BITS-1:0] frm,

    input wire [FLEN-1:0]   dataa,
    input wire [FLEN-1:0]   datab,
    output wire [`VX_CFG_XLEN-1:0]  result,

    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    localparam F32_EXP = 8,  F32_MAN = 23;
    localparam F64_EXP = 11, F64_MAN = 52;
    localparam HAS_D   = (FLEN >= 64);

    `UNUSED_VAR (fmt)
    wire is_d = HAS_D ? fmt[0] : 1'b0;

    // F32 NaN-boxing (RISC-V spec): on a FLEN>32 register file a single-precision
    // operand is valid only if the upper bits are all ones; otherwise the operand
    // is treated as the canonical single-precision qNaN (0x7FC00000).
    localparam [31:0] F32_CANON_QNAN = 32'h7FC00000;
    wire a32_boxed, b32_boxed;
    if (HAS_D) begin : g_box_chk
        assign a32_boxed = &dataa[FLEN-1:32];
        assign b32_boxed = &datab[FLEN-1:32];
    end else begin : g_no_box_chk
        assign a32_boxed = 1'b1;
        assign b32_boxed = 1'b1;
    end
    wire [31:0] a32_op = a32_boxed ? dataa[31:0] : F32_CANON_QNAN;
    wire [31:0] b32_op = b32_boxed ? datab[31:0] : F32_CANON_QNAN;

    reg [LATENCY-1:0] mask_pipe;
    always @(posedge clk) begin
        if (reset) begin
            mask_pipe <= '0;
        end else if (enable) begin
            mask_pipe <= {mask_pipe[LATENCY-2:0], mask};
        end
    end

    localparam  NEG_INF     = 32'h00000001,
                NEG_NORM    = 32'h00000002,
                NEG_SUBNORM = 32'h00000004,
                NEG_ZERO    = 32'h00000008,
                POS_ZERO    = 32'h00000010,
                POS_SUBNORM = 32'h00000020,
                POS_NORM    = 32'h00000040,
                POS_INF     = 32'h00000080,
                //SIG_NAN   = 32'h00000100,
                QUT_NAN     = 32'h00000200;

    // =========================================================================
    // Per-format field extraction, classification and ordering
    //   For each format produce: sign, classification, a<b, a==b.
    //   F32 reads the low 32 bits of the (NaN-boxed) operand.
    // =========================================================================

    // --- F32 (always present) --- operates on the NaN-box-checked operand
    wire        a32_sign = a32_op[31];
    wire        b32_sign = b32_op[31];
    wire [F32_EXP-1:0] a32_exp = a32_op[30:23];
    wire [F32_MAN-1:0] a32_man = a32_op[22:0];
    fclass_t    a32_class, b32_class;

    VX_fp_classifier #(.EXP_BITS(F32_EXP), .MAN_BITS(F32_MAN)) fp_class_a32 (
        .exp_i (a32_exp), .man_i (a32_man), .clss_o (a32_class));
    VX_fp_classifier #(.EXP_BITS(F32_EXP), .MAN_BITS(F32_MAN)) fp_class_b32 (
        .exp_i (b32_op[30:23]), .man_i (b32_op[22:0]), .clss_o (b32_class));

    wire a32_smaller = (a32_op < b32_op) ^ (a32_sign || b32_sign);
    wire ab32_equal  = (a32_op == b32_op)
                    || (a32_class.is_zero && b32_class.is_zero);

    // --- F64 (only when FLEN supports it) ---
    wire        a_sign_sel, b_sign_sel;
    fclass_t    a_class_sel, b_class_sel;
    wire        a_smaller_sel, ab_equal_sel;

    if (HAS_D) begin : g_d_order
        wire        a64_sign = dataa[63];
        wire        b64_sign = datab[63];
        fclass_t    a64_class, b64_class;

        VX_fp_classifier #(.EXP_BITS(F64_EXP), .MAN_BITS(F64_MAN)) fp_class_a64 (
            .exp_i (dataa[62:52]), .man_i (dataa[51:0]), .clss_o (a64_class));
        VX_fp_classifier #(.EXP_BITS(F64_EXP), .MAN_BITS(F64_MAN)) fp_class_b64 (
            .exp_i (datab[62:52]), .man_i (datab[51:0]), .clss_o (b64_class));

        wire a64_smaller = (dataa < datab) ^ (a64_sign || b64_sign);
        wire ab64_equal  = (dataa == datab)
                        || (a64_class.is_zero && b64_class.is_zero);

        assign a_sign_sel    = is_d ? a64_sign    : a32_sign;
        assign b_sign_sel    = is_d ? b64_sign    : b32_sign;
        assign a_class_sel   = is_d ? a64_class   : a32_class;
        assign b_class_sel   = is_d ? b64_class   : b32_class;
        assign a_smaller_sel = is_d ? a64_smaller : a32_smaller;
        assign ab_equal_sel  = is_d ? ab64_equal  : ab32_equal;
    end else begin : g_no_d
        assign a_sign_sel    = a32_sign;
        assign b_sign_sel    = b32_sign;
        assign a_class_sel   = a32_class;
        assign b_class_sel   = b32_class;
        assign a_smaller_sel = a32_smaller;
        assign ab_equal_sel  = ab32_equal;
    end

    // =========================================================================
    // Pipeline stage0 — register operands + ordering primitives
    // =========================================================================

    wire [3:0]      op_mod_s0;
    wire [FLEN-1:0] dataa_s0, datab_s0;
    wire            a_sign_s0, b_sign_s0;
    fclass_t        a_fclass_s0, b_fclass_s0;
    wire            a_smaller_s0, ab_equal_s0;
    wire            is_d_s0;
    wire            a32_boxed_s0;

    wire [3:0] op_mod = {(op_type == INST_FPU_CMP), frm};

    // Raw operands flow to the result stage (FMV.X.W/FMV.X.D copy bits verbatim).
    // a32_boxed is piped so FSGNJ.S can substitute the canonical qNaN payload for
    // a non-NaN-boxed single-precision source. Sign/class/ordering are already
    // derived above from the NaN-box-checked operands.
    VX_pipe_register #(
        .DATAW (4 + 2 * FLEN + 1 + 1 + 2 * $bits(fclass_t) + 1 + 1 + 1 + 1),
        .DEPTH (LATENCY > 0)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask),
        .data_in  ({op_mod,    dataa,    datab,    a_sign_sel, b_sign_sel, a_class_sel, b_class_sel, a_smaller_sel, ab_equal_sel, is_d,    a32_boxed}),
        .data_out ({op_mod_s0, dataa_s0, datab_s0, a_sign_s0,  b_sign_s0,  a_fclass_s0, b_fclass_s0, a_smaller_s0,  ab_equal_s0,  is_d_s0, a32_boxed_s0})
    );

    `UNUSED_VAR (b_fclass_s0)

    // Canonical qNaN per format (NaN-boxed F32 in low 32 when FLEN=64)
    wire [FLEN-1:0] canon_qnan;
    if (HAS_D) begin : g_qnan_d
        assign canon_qnan = is_d_s0 ? {1'b0, {F64_EXP{1'b1}}, 1'b1, {(F64_MAN-1){1'b0}}}
                                    : FLEN'(32'h7FC00000);
    end else begin : g_qnan_s
        assign canon_qnan = FLEN'(32'h7FC00000);
        `UNUSED_VAR (is_d_s0)
    end

    // FCLASS — 10-bit mask in an integer register (format-independent encoding)
    reg [31:0] fclass_mask_s0;
    always @(*) begin
        if (a_fclass_s0.is_normal) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_NORM : POS_NORM;
        end
        else if (a_fclass_s0.is_inf) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_INF : POS_INF;
        end
        else if (a_fclass_s0.is_zero) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_ZERO : POS_ZERO;
        end
        else if (a_fclass_s0.is_subnormal) begin
            fclass_mask_s0 = a_sign_s0 ? NEG_SUBNORM : POS_SUBNORM;
        end
        else if (a_fclass_s0.is_nan) begin
            fclass_mask_s0 = {22'h0, a_fclass_s0.is_quiet, a_fclass_s0.is_signaling, 8'h0};
        end
        else begin
            fclass_mask_s0 = QUT_NAN;
        end
    end

    // Min/Max — returns the surviving operand (raw, format-boxed)
    reg [FLEN-1:0] fminmax_res_s0;
    always @(*) begin
        if (a_fclass_s0.is_nan && b_fclass_s0.is_nan)
            fminmax_res_s0 = canon_qnan;
        else if (a_fclass_s0.is_nan)
            fminmax_res_s0 = datab_s0;
        else if (b_fclass_s0.is_nan)
            fminmax_res_s0 = dataa_s0;
        else begin
            // FMIN, FMAX
            fminmax_res_s0 = (op_mod_s0[0] ^ a_smaller_s0) ? dataa_s0 : datab_s0;
        end
    end

    // Sign injection — replace sign of a, preserve a's exp+man at active width
    reg [FLEN-1:0] fsgnj_res_s0;
    reg sgnj_sign;
    always @(*) begin
        case (op_mod_s0[1:0])
            0:       sgnj_sign =  b_sign_s0;
            1:       sgnj_sign = ~b_sign_s0;
        default:     sgnj_sign =  a_sign_s0 ^ b_sign_s0;
        endcase
    end
    // F32 fp-out lands in the low 32 bits; the result stage boxes the upper
    // half. F64 fills the full width. A non-NaN-boxed F32 source contributes the
    // canonical qNaN payload (its sign is still taken from sgnj_sign above).
    wire [30:0] sgnj_a_payload = a32_boxed_s0 ? dataa_s0[30:0] : F32_CANON_QNAN[30:0];
    if (HAS_D) begin : g_sgnj_d
        assign fsgnj_res_s0 = is_d_s0 ? {sgnj_sign, dataa_s0[62:0]}
                                      : FLEN'({sgnj_sign, sgnj_a_payload});
    end else begin : g_sgnj_s
        assign fsgnj_res_s0 = {sgnj_sign, sgnj_a_payload};
    end

    // Comparison
    reg fcmp_res_s0;
    reg fcmp_fflags_NV_s0;
    always @(*) begin
        case (op_mod_s0[1:0])
            0: begin // LE
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = 1;
                end else begin
                    fcmp_res_s0       = (a_smaller_s0 | ab_equal_s0);
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            1: begin // LT
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = 1;
                end else begin
                    fcmp_res_s0       = (a_smaller_s0 & ~ab_equal_s0);
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            2: begin // EQ
                if (a_fclass_s0.is_nan || b_fclass_s0.is_nan) begin
                    fcmp_res_s0       = 0;
                    fcmp_fflags_NV_s0 = a_fclass_s0.is_signaling | b_fclass_s0.is_signaling;
                end else begin
                    fcmp_res_s0       = ab_equal_s0;
                    fcmp_fflags_NV_s0 = 0;
                end
            end
            default: begin
                fcmp_res_s0       = 'x;
                fcmp_fflags_NV_s0 = 'x;
            end
        endcase
    end

    // Op result mux (float-domain width = FLEN; integer ops in low bits)
    reg [FLEN-1:0] result_s0;
    reg fflags_NV_s0;
    always @(*) begin
        case (op_mod_s0[2:0])
            0,1,2: begin
                // SGNJ, CMP
                result_s0 = op_mod_s0[3] ? FLEN'(fcmp_res_s0) : fsgnj_res_s0;
                fflags_NV_s0 = fcmp_fflags_NV_s0;
            end
            3: begin
                // CLASS
                result_s0 = FLEN'(fclass_mask_s0);
                fflags_NV_s0 = 0;
            end
            4,5: begin
                // FMV
                result_s0 = dataa_s0;
                fflags_NV_s0 = 0;
            end
            6,7: begin
                // MIN/MAX
                result_s0 = fminmax_res_s0;
                fflags_NV_s0 = a_fclass_s0.is_signaling | b_fclass_s0.is_signaling;
            end
        endcase
    end

    // Extend/box result to XLEN:
    //   CMP/CLASS (integer output)  → zero-extend
    //   MVXW      (float → int reg) → sign-extend (FMV.X.W); FMV.X.D passes 64b
    //   SGNJ/MVWX/MIN/MAX (FP out)  → F64 passes 64b; F32 NaN-boxes upper bits
    //                                 (F32 fp results already carry the box)
    wire [`VX_CFG_XLEN-1:0] result_xlen_s0;
    if (`VX_CFG_XLEN > 32) begin : g_result_ext
        wire is_int_result = op_mod_s0[3]              // CMP
                          || (op_mod_s0[2:0] == 3'd3); // CLASS
        wire is_mvxw      = (op_mod_s0[2:0] == 3'd4) && !op_mod_s0[3];
        assign result_xlen_s0 = is_int_result ? `VX_CFG_XLEN'(result_s0[31:0])
                              : is_d_s0       ? `VX_CFG_XLEN'(result_s0) // F64 fp / FMV.X.D
                              : is_mvxw       ? {{(`VX_CFG_XLEN-32){result_s0[31]}}, result_s0[31:0]}
                              :                 {{(`VX_CFG_XLEN-32){1'b1}}, result_s0[31:0]}; // F32 fp NaN-box
    end else begin : g_no_result_ext
        assign result_xlen_s0 = result_s0;
    end

    wire fflags_NV;

    VX_pipe_register #(
        .DATAW (`VX_CFG_XLEN + 1),
        .DEPTH (OUT_REG)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable && mask_pipe[LATENCY-2]),
        .data_in  ({result_xlen_s0, fflags_NV_s0}),
        .data_out ({result,         fflags_NV})
    );
                    // NV,      DZ,   OF,   UF,   NX
    assign fflags = {fflags_NV, 1'b0, 1'b0, 1'b0, 1'b0};

endmodule
