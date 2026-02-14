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

module VX_tcu_drl_exp_bias import VX_tcu_pkg::*;  #(
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]                fmtf,

    // Raw Inputs
    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,
    input wire [31:0]               c_val,

    // Classification Inputs
    input fedp_class_t [N-1:0]      cls_tf32 [2],
    input fedp_class_t [TCK-1:0]    cls_fp16 [2],
    input fedp_class_t [TCK-1:0]    cls_bf16 [2],
    input fedp_class_t [2*TCK-1:0]  cls_fp8 [2],
    input fedp_class_t [2*TCK-1:0]  cls_bf8 [2],
    input fedp_class_t              cls_c,

    // Output increased to [TCK:0] to include C-term
    output wire [TCK:0][EXP_W-1:0]  raw_exp_y,
    output wire [TCK-1:0][5:0]      exp_diff_f8
);
    `UNUSED_VAR({vld_mask})

    localparam F32_BIAS     = 127;
    localparam F32_SIG_BITS = 23;
    localparam MUL_WIDTH    = 22;
    localparam ALIGN_SHIFT  = F32_SIG_BITS - MUL_WIDTH; // +1

    localparam E_TF32 = VX_tcu_pkg::exp_bits(TCU_TF32_ID);
    localparam S_TF32 = VX_tcu_pkg::sign_pos(TCU_TF32_ID);
    localparam B_TF32 = (1 << (E_TF32 - 1)) - 1;

    localparam E_FP16 = VX_tcu_pkg::exp_bits(TCU_FP16_ID);
    localparam S_FP16 = VX_tcu_pkg::sign_pos(TCU_FP16_ID);
    localparam B_FP16 = (1 << (E_FP16 - 1)) - 1;

    localparam E_BF16 = VX_tcu_pkg::exp_bits(TCU_BF16_ID);
    localparam S_BF16 = VX_tcu_pkg::sign_pos(TCU_BF16_ID);
    localparam B_BF16 = (1 << (E_BF16 - 1)) - 1;

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1 ;

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1 ;

    localparam [7:0] BIAS_CONST_TF32 = 8'(F32_BIAS - 2*B_TF32 + ALIGN_SHIFT   - W + WA - 1);
    localparam [7:0] BIAS_CONST_FP16 = 8'(F32_BIAS - 2*B_FP16 + ALIGN_SHIFT   - W + WA - 1);
    localparam [7:0] BIAS_CONST_BF16 = 8'(F32_BIAS - 2*B_BF16 + ALIGN_SHIFT   - W + WA - 1);
    localparam [7:0] BIAS_CONST_FP8  = 8'(F32_BIAS - 2*B_FP8  + 2*ALIGN_SHIFT - W + WA - 1);
    localparam [7:0] BIAS_CONST_BF8  = 8'(F32_BIAS - 2*B_BF8  + 2*ALIGN_SHIFT - W + WA - 1);
    localparam [EXP_W-1:0] EXP_NEG_INF = {1'b1, {(EXP_W-1){1'b0}}};

    `UNUSED_PARAM (BIAS_CONST_TF32)
    `UNUSED_PARAM (BIAS_CONST_BF16)
    `UNUSED_PARAM (BIAS_CONST_FP8)
    `UNUSED_PARAM (BIAS_CONST_BF8)

    // ----------------------------------------------------------------------
    // 1. Inputs Setup
    // ----------------------------------------------------------------------

    // --- TF32 Preparation ---
    wire [TCK-1:0][7:0] ea_tf32, eb_tf32;
    wire [TCK-1:0]      z_tf32;
    `UNUSED_VAR ({ea_tf32, eb_tf32, z_tf32})
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin : g_even_lane
            assign ea_tf32[i] = cls_tf32[0][i/2].is_sub ? 8'd1 : a_row[i/2][S_TF32-1 -: E_TF32];
            assign eb_tf32[i] = cls_tf32[1][i/2].is_sub ? 8'd1 : b_col[i/2][S_TF32-1 -: E_TF32];
            assign z_tf32[i]  = cls_tf32[0][i/2].is_zero | cls_tf32[1][i/2].is_zero | ~vld_mask[i*4];
        end else begin : g_odd_lane
            assign ea_tf32[i] = 8'd0;
            assign eb_tf32[i] = 8'd0;
            assign z_tf32[i]  = 1'b1;
        end
    end

    // --- FP16 Preparation ---
    wire [TCK-1:0][4:0] ea_fp16, eb_fp16;
    wire [TCK-1:0]      z_fp16;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp16
        localparam OFF = (i % 2) * 16;
        assign ea_fp16[i] = cls_fp16[0][i].is_sub ? 5'd1 : a_row[i/2][S_FP16-1+OFF -: E_FP16];
        assign eb_fp16[i] = cls_fp16[1][i].is_sub ? 5'd1 : b_col[i/2][S_FP16-1+OFF -: E_FP16];
        assign z_fp16[i]  = cls_fp16[0][i].is_zero | cls_fp16[1][i].is_zero | ~vld_mask[i*4];
    end

    // --- BF16 Preparation ---
    wire [TCK-1:0][7:0] ea_bf16, eb_bf16;
    wire [TCK-1:0]      z_bf16;
    `UNUSED_VAR ({ea_bf16, eb_bf16, z_bf16})
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf16
        localparam OFF = (i % 2) * 16;
        assign ea_bf16[i] = cls_bf16[0][i].is_sub ? 8'd1 : a_row[i/2][S_BF16-1+OFF -: E_BF16];
        assign eb_bf16[i] = cls_bf16[1][i].is_sub ? 8'd1 : b_col[i/2][S_BF16-1+OFF -: E_BF16];
        assign z_bf16[i]  = cls_bf16[0][i].is_zero | cls_bf16[1][i].is_zero | ~vld_mask[i*4];
    end

    // --- FP8 Preparation ---
    wire [TCK-1:0][1:0][3:0] ea_fp8, eb_fp8;
    wire [TCK-1:0][1:0]      z_fp8;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            localparam OFF = (i % 2) * 16 + j * 8;
            assign ea_fp8[i][j] = cls_fp8[0][idx].is_sub ? 4'd1 : a_row[i/2][S_FP8-1+OFF -: E_FP8];
            assign eb_fp8[i][j] = cls_fp8[1][idx].is_sub ? 4'd1 : b_col[i/2][S_FP8-1+OFF -: E_FP8];
            assign z_fp8[i][j]  = cls_fp8[0][idx].is_zero | cls_fp8[1][idx].is_zero | ~vld_mask[idx*2];
        end
    end

    // --- BF8 Preparation ---
    wire [TCK-1:0][1:0][4:0] ea_bf8, eb_bf8;
    wire [TCK-1:0][1:0]      z_bf8;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            localparam OFF = (i % 2) * 16 + j * 8;
            assign ea_bf8[i][j] = cls_bf8[0][idx].is_sub ? 5'd1 : a_row[i/2][S_BF8-1+OFF -: E_BF8];
            assign eb_bf8[i][j] = cls_bf8[1][idx].is_sub ? 5'd1 : b_col[i/2][S_BF8-1+OFF -: E_BF8];
            assign z_bf8[i][j]  = cls_bf8[0][idx].is_zero | cls_bf8[1][idx].is_zero | ~vld_mask[idx*2];
        end
    end

    // ----------------------------------------------------------------------
    // 2. Product Computation
    // ----------------------------------------------------------------------

        // f16 Mux Selection
    wire  [TCK-1:0][EXP_W-1:0] sum_f16;
    logic [TCK-1:0]            is_zero_f16;
    for (genvar i = 0; i < TCK; ++i) begin : g_exp_f16
        logic [7:0] ea_sel, eb_sel;
        logic [7:0] bias_sel;
        logic       is_zero;

        always_comb begin
            case (fmtf)
                TCU_FP16_ID: begin
                    ea_sel   = 8'(ea_fp16[i]);
                    eb_sel   = 8'(eb_fp16[i]);
                    is_zero  = z_fp16[i];
                    bias_sel = BIAS_CONST_FP16;
                end
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: begin
                    ea_sel   = 8'(ea_bf16[i]);
                    eb_sel   = 8'(eb_bf16[i]);
                    is_zero  = z_bf16[i];
                    bias_sel = BIAS_CONST_BF16;
                end
            `endif
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID: begin
                    ea_sel   = 8'(ea_tf32[i]);
                    eb_sel   = 8'(eb_tf32[i]);
                    is_zero  = z_tf32[i];
                    bias_sel = BIAS_CONST_TF32;
                end
            `endif
                default: begin
                    ea_sel  = 'x;
                    eb_sel  = 'x;
                    is_zero = 'x;
                    bias_sel= 'x;
                end
            endcase
        end

        VX_csa_tree #(
            .N (3),
            .W (EXP_W),
            .S (EXP_W),
            .CPA_KS (!`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_adder_f16 (
            .operands ({EXP_W'($signed(bias_sel)), EXP_W'(ea_sel), EXP_W'(eb_sel)}),
            .sum      (sum_f16[i]),
            `UNUSED_PIN (cout)
        );
        assign is_zero_f16[i] = is_zero;
    end

    // f8 Mux Selection
    wire [TCK-1:0][EXP_W-1:0] sum_f8_0, sum_f8_1;
    wire [TCK-1:0]            diff_sign_f8;
    logic [TCK-1:0][1:0]      is_zero_f8;
    `UNUSED_VAR ({diff_sign_f8, is_zero_f8, sum_f8_0, sum_f8_1})
    for (genvar i = 0; i < TCK; ++i) begin : g_exp_f8
        logic [1:0][4:0] ea_sel, eb_sel;
        logic [7:0]      bias_sel;
        logic [1:0]      is_zero;
        always_comb begin
            case (fmtf)
                TCU_FP8_ID: begin
                    ea_sel[0] = 5'(ea_fp8[i][0]);
                    ea_sel[1] = 5'(ea_fp8[i][1]);
                    eb_sel[0] = 5'(eb_fp8[i][0]);
                    eb_sel[1] = 5'(eb_fp8[i][1]);
                    is_zero   = z_fp8[i];
                    bias_sel  = BIAS_CONST_FP8;
                end
                TCU_BF8_ID: begin
                    ea_sel[0] = 5'(ea_bf8[i][0]);
                    ea_sel[1] = 5'(ea_bf8[i][1]);
                    eb_sel[0] = 5'(eb_bf8[i][0]);
                    eb_sel[1] = 5'(eb_bf8[i][1]);
                    is_zero   = z_bf8[i];
                    bias_sel  = BIAS_CONST_BF8;
                end
                default: begin
                    ea_sel    = 'x;
                    eb_sel    = 'x;
                    bias_sel  = 'x;
                    is_zero   = 'x;
                end
            endcase
        end

        wire [5:0] pre_sum_0, pre_sum_1;
        VX_ks_adder #(
            .N (6),
            .BYPASS (`FORCE_BUILTIN_ADDER(6))
        ) exp_adder1_f8_0 (
            .dataa (6'(ea_sel[0])),
            .datab (6'(eb_sel[0])),
            .cin   (1'b0),
            .sum   (pre_sum_0),
            `UNUSED_PIN (cout)
        );

        VX_ks_adder #(
            .N (6),
            .BYPASS (`FORCE_BUILTIN_ADDER(6))
        ) exp_adder1_f8_1 (
            .dataa (6'(ea_sel[1])),
            .datab (6'(eb_sel[1])),
            .cin   (1'b0),
            .sum   (pre_sum_1),
            `UNUSED_PIN (cout)
        );

        VX_ks_adder #(
            .N (EXP_W),
            .BYPASS (`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_adder2_f8_0 (
            .dataa (EXP_W'(pre_sum_0)),
            .datab (EXP_W'(bias_sel)),
            .cin   (1'b0),
            .sum   (sum_f8_0[i]),
            `UNUSED_PIN (cout)
        );

        VX_ks_adder #(
            .N (EXP_W),
            .BYPASS (`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_adder2_f8_1 (
            .dataa (EXP_W'(pre_sum_1)),
            .datab (EXP_W'(bias_sel)),
            .cin   (1'b0),
            .sum   (sum_f8_1[i]),
            `UNUSED_PIN (cout)
        );

        // Difference calculation for alignment
        wire diff_sign = (pre_sum_1 < pre_sum_0);
        wire [5:0] max_sum = diff_sign ? pre_sum_0 : pre_sum_1;
        wire [5:0] min_sum = diff_sign ? pre_sum_1 : pre_sum_0;
        wire [5:0] diff_abs;
        VX_ks_adder #(
            .N (6),
            .BYPASS (`FORCE_BUILTIN_ADDER(6))
        ) ks_diff_f8 (
            .dataa (max_sum),
            .datab (~min_sum),
            .cin   (1'b1),
            .sum   (diff_abs),
            `UNUSED_PIN (cout)
        );
        `UNUSED_VAR (diff_abs[5])
        assign exp_diff_f8[i]  = {diff_sign, diff_abs[4:0]};
        assign diff_sign_f8[i] = diff_sign;
        assign is_zero_f8[i]   = is_zero;
    end

        // Final Output Mux
    for (genvar i = 0; i < TCK; ++i) begin : g_exp_mux
        logic [EXP_W-1:0] prod_exp;
        always_comb begin
            case(fmtf)
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID,
            `endif
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID,
            `endif
                TCU_FP16_ID: begin
                    prod_exp = is_zero_f16[i] ? EXP_NEG_INF : sum_f16[i];
                end
            `ifdef TCU_FP8_ENABLE
                TCU_FP8_ID, TCU_BF8_ID: begin
                    prod_exp = diff_sign_f8[i] ?
                        (is_zero_f8[i][0] ? EXP_NEG_INF : sum_f8_0[i]) :
                        (is_zero_f8[i][1] ? EXP_NEG_INF : sum_f8_1[i]);
                end
            `endif
                default: begin
                    prod_exp = 'x;
                end
            endcase
        end
        assign raw_exp_y[i] = prod_exp;
    end

    // ----------------------------------------------------------------------
    // 3. C-Term Exponent
    // ----------------------------------------------------------------------

    // Corrected to include Window Adjustment: c_exp - (W - 1) + WA - 1
    `UNUSED_VAR ({c_val[31], c_val[23:0], cls_c})
    wire [7:0] c_exp_raw = c_val[30:23];
    wire [EXP_W-1:0] c_exp_adj = EXP_W'(c_exp_raw) - EXP_W'(W-1) + EXP_W'(WA-1);
    assign raw_exp_y[TCK] = cls_c.is_zero ? EXP_NEG_INF : c_exp_adj;

endmodule
