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

`include "VX_define.vh"

module VX_tcu_tfr_mul_f16 import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10
) (
    input wire                      clk,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]                fmt_f,

    // Raw Inputs (No pre-classification)
    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,

    // Outputs
    output logic [TCK-1:0][24:0]      result_sig,
    output logic [TCK-1:0][EXP_W-1:0] result_exp,
    output fedp_excep_t [TCK-1:0]     exceptions
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})
    `UNUSED_VAR (vld_mask)

    // ======================================================================
    // 1. Internal Classification (Replicated Logic)
    // ======================================================================

    fedp_class_t [N-1:0]   cls_tf32 [2];
    fedp_class_t [TCK-1:0] cls_fp16 [2];
    fedp_class_t [TCK-1:0] cls_bf16 [2];

    VX_tcu_tfr_classifier #(.N(N), .WIDTH(32), .FMT(TCU_TF32_ID)) c_a_tf32 (.val(a_row), .cls(cls_tf32[0]));
    VX_tcu_tfr_classifier #(.N(N), .WIDTH(32), .FMT(TCU_TF32_ID)) c_b_tf32 (.val(b_col), .cls(cls_tf32[1]));

    VX_tcu_tfr_classifier #(.N(TCK), .WIDTH(16), .FMT(TCU_FP16_ID)) c_a_fp16 (.val(a_row), .cls(cls_fp16[0]));
    VX_tcu_tfr_classifier #(.N(TCK), .WIDTH(16), .FMT(TCU_FP16_ID)) c_b_fp16 (.val(b_col), .cls(cls_fp16[1]));

    VX_tcu_tfr_classifier #(.N(TCK), .WIDTH(16), .FMT(TCU_BF16_ID)) c_a_bf16 (.val(a_row), .cls(cls_bf16[0]));
    VX_tcu_tfr_classifier #(.N(TCK), .WIDTH(16), .FMT(TCU_BF16_ID)) c_b_bf16 (.val(b_col), .cls(cls_bf16[1]));

    // ======================================================================
    // 2. Constants & Parameters
    // ======================================================================

    localparam F32_BIAS  = 127;
    localparam S_FP32    = 23;
    localparam S_SUPER   = 22;

    // adding +128 to bias base to ensure BIAS in [0..255] range
    localparam BIAS_BASE = F32_BIAS + S_FP32 - S_SUPER - W + WA - 1 + 128;

    localparam E_TF32 = VX_tcu_pkg::exp_bits(TCU_TF32_ID);
    localparam S_TF32 = VX_tcu_pkg::sign_pos(TCU_TF32_ID);
    localparam B_TF32 = (1 << (E_TF32 - 1)) - 1;
    localparam [7:0] BIAS_CONST_TF32 = 8'(BIAS_BASE - 2*B_TF32);

    localparam E_FP16 = VX_tcu_pkg::exp_bits(TCU_FP16_ID);
    localparam S_FP16 = VX_tcu_pkg::sign_pos(TCU_FP16_ID);
    localparam B_FP16 = (1 << (E_FP16 - 1)) - 1;
    localparam [7:0] BIAS_CONST_FP16 = 8'(BIAS_BASE - 2*B_FP16);

    localparam E_BF16 = VX_tcu_pkg::exp_bits(TCU_BF16_ID);
    localparam S_BF16 = VX_tcu_pkg::sign_pos(TCU_BF16_ID);
    localparam B_BF16 = (1 << (E_BF16 - 1)) - 1;
    localparam [7:0] BIAS_CONST_BF16 = 8'(BIAS_BASE - 2*B_BF16);

    `UNUSED_PARAM (BIAS_CONST_TF32)
    `UNUSED_PARAM (BIAS_CONST_BF16)

    // ======================================================================
    // 3. Main Loop (Per TCK Lane)
    // ======================================================================

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        // Selected Signals (Mux Output)
        logic [7:0]  ea_sel, eb_sel;
        logic [7:0]  bias_sel;
        logic [10:0] ma_sel, mb_sel;
        logic        zero_sel, sign_sel, nan_sel, inf_sel;

        // ------------------------------------------------------------------
        // 3a. Pre-Calculation (Independent Paths)
        // ------------------------------------------------------------------

        // Lane Masking
        wire lane_valid = vld_mask[i*4];
        localparam OFF_16 = (i % 2) * 16;

        // --- FP16 Signals -------------------------------------------------
        wire [7:0]  ea_fp16 = cls_fp16[0][i].is_sub ? 8'b1 : 8'(a_row[i/2][S_FP16-1+OFF_16 -: E_FP16]);
        wire [7:0]  eb_fp16 = cls_fp16[1][i].is_sub ? 8'b1 : 8'(b_col[i/2][S_FP16-1+OFF_16 -: E_FP16]);
        wire        z_fp16  = cls_fp16[0][i].is_zero | cls_fp16[1][i].is_zero;

        wire [10:0] ma_fp16 = cls_fp16[0][i].is_zero ? 11'b0 : {!cls_fp16[0][i].is_sub, a_row[i/2][9+OFF_16 -: 10]};
        wire [10:0] mb_fp16 = cls_fp16[1][i].is_zero ? 11'b0 : {!cls_fp16[1][i].is_sub, b_col[i/2][9+OFF_16 -: 10]};
        wire        s_fp16  = cls_fp16[0][i].sign ^ cls_fp16[1][i].sign;

        wire nan_in_fp16 = cls_fp16[0][i].is_nan | cls_fp16[1][i].is_nan;
        wire inf_z_fp16  = (cls_fp16[0][i].is_inf & cls_fp16[1][i].is_zero) |
                           (cls_fp16[0][i].is_zero & cls_fp16[1][i].is_inf);
        wire inf_op_fp16 = cls_fp16[0][i].is_inf | cls_fp16[1][i].is_inf;

        wire nan_fp16 = nan_in_fp16 | inf_z_fp16;
        wire inf_fp16 = inf_op_fp16 & ~inf_z_fp16;

        // --- BF16 Signals -------------------------------------------------
        wire [7:0]  ea_bf16 = cls_bf16[0][i].is_sub ? 8'b1 : a_row[i/2][S_BF16-1+OFF_16 -: E_BF16];
        wire [7:0]  eb_bf16 = cls_bf16[1][i].is_sub ? 8'b1 : b_col[i/2][S_BF16-1+OFF_16 -: E_BF16];
        wire        z_bf16  = cls_bf16[0][i].is_zero | cls_bf16[1][i].is_zero;

        wire [10:0] ma_bf16 = cls_bf16[0][i].is_zero ? 11'b0 : {3'b0, !cls_bf16[0][i].is_sub, a_row[i/2][6+OFF_16 -: 7]};
        wire [10:0] mb_bf16 = cls_bf16[1][i].is_zero ? 11'b0 : {3'b0, !cls_bf16[1][i].is_sub, b_col[i/2][6+OFF_16 -: 7]};
        wire        s_bf16  = cls_bf16[0][i].sign ^ cls_bf16[1][i].sign;

        wire nan_in_bf16 = cls_bf16[0][i].is_nan | cls_bf16[1][i].is_nan;
        wire inf_z_bf16  = (cls_bf16[0][i].is_inf & cls_bf16[1][i].is_zero) |
                           (cls_bf16[0][i].is_zero & cls_bf16[1][i].is_inf);
        wire inf_op_bf16 = cls_bf16[0][i].is_inf | cls_bf16[1][i].is_inf;

        wire nan_bf16 = nan_in_bf16 | inf_z_bf16;
        wire inf_bf16 = inf_op_bf16 & ~inf_z_bf16;
        `UNUSED_VAR ({ea_bf16, eb_bf16, ma_bf16, mb_bf16, s_bf16, z_bf16, nan_bf16, inf_bf16})

        // --- TF32 Signals -------------------------------------------------
        wire [7:0]  ea_tf32, eb_tf32;
        wire [10:0] ma_tf32, mb_tf32;
        wire        z_tf32, s_tf32, nan_tf32, inf_tf32;

        if ((i % 2) == 0) begin : g_tf32_even
            assign ea_tf32 = cls_tf32[0][i/2].is_sub ? 8'b1 : a_row[i/2][S_TF32-1 -: E_TF32];
            assign eb_tf32 = cls_tf32[1][i/2].is_sub ? 8'b1 : b_col[i/2][S_TF32-1 -: E_TF32];
            assign z_tf32  = cls_tf32[0][i/2].is_zero | cls_tf32[1][i/2].is_zero;

            assign ma_tf32 = cls_tf32[0][i/2].is_zero ? 11'b0 : {!cls_tf32[0][i/2].is_sub, a_row[i/2][9:0]};
            assign mb_tf32 = cls_tf32[1][i/2].is_zero ? 11'b0 : {!cls_tf32[1][i/2].is_sub, b_col[i/2][9:0]};
            assign s_tf32  = cls_tf32[0][i/2].sign ^ cls_tf32[1][i/2].sign;

            wire nan_in_tf32 = cls_tf32[0][i/2].is_nan | cls_tf32[1][i/2].is_nan;
            wire inf_z_tf32  = (cls_tf32[0][i/2].is_inf & cls_tf32[1][i/2].is_zero) |
                               (cls_tf32[0][i/2].is_zero & cls_tf32[1][i/2].is_inf);
            wire inf_op_tf32 = cls_tf32[0][i/2].is_inf | cls_tf32[1][i/2].is_inf;

            assign nan_tf32 = nan_in_tf32 | inf_z_tf32;
            assign inf_tf32 = inf_op_tf32 & ~inf_z_tf32;
        end else begin : g_tf32_odd
            assign ea_tf32  = '0;
            assign eb_tf32  = '0;
            assign ma_tf32  = '0;
            assign mb_tf32  = '0;
            assign z_tf32   = 1'b1;
            assign s_tf32   = 1'b0;
            assign nan_tf32 = 1'b0;
            assign inf_tf32 = 1'b0;
        end
        `UNUSED_VAR ({ea_tf32, eb_tf32, ma_tf32, mb_tf32, s_tf32, z_tf32, nan_tf32, inf_tf32})

        // ------------------------------------------------------------------
        // 3b. Muxing (Selection Only)
        // ------------------------------------------------------------------
        always_comb begin
            case(fmt_f)
                TCU_FP16_ID: begin
                    ea_sel      = ea_fp16;
                    eb_sel      = eb_fp16;
                    ma_sel      = ma_fp16;
                    mb_sel      = mb_fp16;
                    bias_sel    = BIAS_CONST_FP16;
                    sign_sel    = s_fp16;
                    zero_sel    = z_fp16;
                    nan_sel     = nan_fp16;
                    inf_sel     = inf_fp16;
                end
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: begin
                    ea_sel      = ea_bf16;
                    eb_sel      = eb_bf16;
                    ma_sel      = ma_bf16;
                    mb_sel      = mb_bf16;
                    bias_sel    = BIAS_CONST_BF16;
                    sign_sel    = s_bf16;
                    zero_sel    = z_bf16;
                    nan_sel     = nan_bf16;
                    inf_sel     = inf_bf16;
                end
            `endif
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID: begin
                    ea_sel      = ea_tf32;
                    eb_sel      = eb_tf32;
                    ma_sel      = ma_tf32;
                    mb_sel      = mb_tf32;
                    bias_sel    = BIAS_CONST_TF32;
                    sign_sel    = s_tf32;
                    zero_sel    = z_tf32;
                    nan_sel     = nan_tf32;
                    inf_sel     = inf_tf32;
                end
            `endif
                default: begin
                    ea_sel      = 'x;
                    eb_sel      = 'x;
                    ma_sel      = 'x;
                    mb_sel      = 'x;
                    bias_sel    = 'x;
                    sign_sel    = 'x;
                    zero_sel    = 'x;
                    nan_sel     = 'x;
                    inf_sel     = 'x;
                end
            endcase
        end

        // ------------------------------------------------------------------
        // 3c. Arithmetic Path
        // ------------------------------------------------------------------

        // Exponent Addition

        wire [EXP_W-1:0] exp_sum, exp_carry;
        VX_csa_tree #(
            .N (3),
            .W (EXP_W),
            .S (EXP_W)
        ) exp_csa (
            .operands ({EXP_W'(bias_sel), EXP_W'(ea_sel), EXP_W'(eb_sel)}),
            .sum      (exp_sum),
            .carry    (exp_carry)
        );

        wire [EXP_W-1:0] exp_final_sum;
        VX_ks_adder #(
            .N (EXP_W),
            .BYPASS (!`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_ksa (
            .cin   (1'b0),
            .dataa (exp_sum),
            .datab (exp_carry),
            .sum   (exp_final_sum),
            `UNUSED_PIN(cout)
        );

        assign result_exp[i] = (~zero_sel && lane_valid) ? exp_final_sum : '0;

        // Mantissa Multiplication
        wire [21:0] man_prod;
        VX_wallace_mul #(
            .N (11),
            .CPA_KS (!`FORCE_BUILTIN_ADDER(11*2))
        ) wtmul (
            .a(ma_sel),
            .b(mb_sel),
            .p(man_prod)
        );

        // Output Formatting
        always_comb begin
            case (fmt_f)
                TCU_FP16_ID: result_sig[i] = {sign_sel, man_prod, 2'b0};
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: result_sig[i] = {sign_sel, man_prod[15:0], 8'b0};
            `endif
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID: result_sig[i] = {sign_sel, man_prod, 2'b0};
            `endif
                default:     result_sig[i] = 'x;
            endcase
        end

        // Exception Outputs
        assign exceptions[i].is_nan = nan_sel && lane_valid;
        assign exceptions[i].is_inf = inf_sel && lane_valid;
        assign exceptions[i].sign   = sign_sel;
    end

endmodule
