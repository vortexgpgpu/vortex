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

module VX_tcu_tfr_mul_f16 import VX_tcu_pkg::*;
#(
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
    // 1. Constants & Parameters
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
    `UNUSED_PARAM (S_TF32)
    `UNUSED_PARAM (S_BF16)


    // ======================================================================
    // 2. Main Loop (Per TCK Lane)
    // ======================================================================

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire lane_valid = vld_mask[i*4];
        localparam OFF_16 = (i % 2) * 16;

        // ------------------------------------------------------------------
        // 2a. Input Muxing & Field Extraction
        // ------------------------------------------------------------------
        logic [7:0] raw_ea, raw_eb;
        logic [9:0] raw_ma, raw_mb;
        logic       raw_sa, raw_sb;
        logic [7:0] exp_max;
        logic [7:0] bias_sel;

        always_comb begin
            case (fmt_f)
                TCU_FP16_ID: begin
                    raw_ea    = 8'(a_row[i/2][S_FP16-1+OFF_16 -: E_FP16]);
                    raw_eb    = 8'(b_col[i/2][S_FP16-1+OFF_16 -: E_FP16]);
                    raw_ma    = a_row[i/2][9+OFF_16 -: 10];
                    raw_mb    = b_col[i/2][9+OFF_16 -: 10];
                    raw_sa    = a_row[i/2][15+OFF_16];
                    raw_sb    = b_col[i/2][15+OFF_16];
                    exp_max   = 8'h1F;
                    bias_sel  = BIAS_CONST_FP16;
                end
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: begin
                    raw_ea    = a_row[i/2][S_BF16-1+OFF_16 -: E_BF16];
                    raw_eb    = b_col[i/2][S_BF16-1+OFF_16 -: E_BF16];
                    raw_ma    = {a_row[i/2][6+OFF_16 -: 7], 3'b0};
                    raw_mb    = {b_col[i/2][6+OFF_16 -: 7], 3'b0};
                    raw_sa    = a_row[i/2][15+OFF_16];
                    raw_sb    = b_col[i/2][15+OFF_16];
                    exp_max   = 8'hFF;
                    bias_sel  = BIAS_CONST_BF16;
                end
            `endif
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID: begin
                    if ((i % 2) == 0) begin
                        raw_ea    = a_row[i/2][S_TF32-1 -: E_TF32];
                        raw_eb    = b_col[i/2][S_TF32-1 -: E_TF32];
                        raw_ma    = a_row[i/2][9:0];
                        raw_mb    = b_col[i/2][9:0];
                        raw_sa    = a_row[i/2][18];
                        raw_sb    = b_col[i/2][18];
                        exp_max   = 8'hFF;
                        bias_sel  = BIAS_CONST_TF32;
                    end else begin
                        raw_ea    = '0;
                        raw_eb    = '0;
                        raw_ma    = '0;
                        raw_mb    = '0;
                        raw_sa    = '0;
                        raw_sb    = '0;
                        exp_max   = 8'hFF;
                        bias_sel  = '0;
                    end
                end
            `endif
                default: begin
                    raw_ea    = 'x;
                    raw_eb    = 'x;
                    raw_ma    = 'x;
                    raw_mb    = 'x;
                    raw_sa    = 'x;
                    raw_sb    = 'x;
                    exp_max   = 'x;
                    bias_sel  = 'x;
                end
            endcase
        end

        // ------------------------------------------------------------------
        // 2b. Inline Generic Classification
        // ------------------------------------------------------------------

        fedp_class_t cls_a;
        VX_tcu_tfr_classifier #(
            .EXP_W (8),
            .MAN_W (10)
        ) class_a (
            .exp (raw_ea),
            .man (raw_ma),
            .max_exp (exp_max),
            .cls (cls_a)
        );

        fedp_class_t cls_b;
        VX_tcu_tfr_classifier #(
            .EXP_W (8),
            .MAN_W (10)
        ) class_b (
            .exp (raw_eb),
            .man (raw_mb),
            .max_exp (exp_max),
            .cls (cls_b)
        );

        // ------------------------------------------------------------------
        // 2c. Operand Preparation
        // ------------------------------------------------------------------
        wire [7:0] ea_sel = cls_a.is_sub ? 8'b1 : raw_ea;
        wire [7:0] eb_sel = cls_b.is_sub ? 8'b1 : raw_eb;

        wire [10:0] ma_sel = {!cls_a.is_sub, raw_ma};
        wire [10:0] mb_sel = {!cls_b.is_sub, raw_mb};

        wire sign_sel = raw_sa ^ raw_sb;
        wire zero_sel = cls_a.is_zero | cls_b.is_zero;

        wire nan_in = cls_a.is_nan | cls_b.is_nan;
        wire inf_z  = (cls_a.is_inf & cls_b.is_zero) | (cls_a.is_zero & cls_b.is_inf);
        wire inf_op = cls_a.is_inf | cls_b.is_inf;

        wire nan_sel = nan_in | inf_z;
        wire inf_sel = inf_op & ~inf_z;

        // ------------------------------------------------------------------
        // 2d. Arithmetic Path
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
                TCU_BF16_ID: result_sig[i] = {sign_sel, man_prod, 2'b0};
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
