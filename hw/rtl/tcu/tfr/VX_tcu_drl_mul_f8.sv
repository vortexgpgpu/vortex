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

module VX_tcu_tfr_mul_f8 import VX_tcu_pkg::*;
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
    // f8 lanes need 2*ALIGN_SHIFT (one per sub-product) unlike f16 which needs 1x
    localparam BIAS_BASE = F32_BIAS + 2*(S_FP32 - S_SUPER) - W + WA - 1 + 128;

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam S_FP8 = VX_tcu_pkg::sign_pos(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_FP8  = 8'(BIAS_BASE - 2*B_FP8);

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam S_BF8 = VX_tcu_pkg::sign_pos(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_BF8  = 8'(BIAS_BASE - 2*B_BF8);

    wire is_bfloat = tcu_fmt_is_bfloat(fmt_f);

    // ======================================================================
    // 2. Main Loop (Per TCK Lane)
    // ======================================================================

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        // Per-element valid bits (2 elements -> 1 lane)
        // 8-bit element k validity is stored at vld_mask[2*k].
        wire [1:0] lane_valid = {vld_mask[i * 4 + 2], vld_mask[i * 4 + 0]};

        // ------------------------------------------------------------------
        // 2a. Pre-Calculation & Inline Classification
        // ------------------------------------------------------------------

        wire [1:0][4:0] ea_sel, eb_sel;
        wire [1:0][3:0] ma_sel, mb_sel;
        wire [1:0]      zero_sel, sign_sel, nan_sel, inf_sel;

        for (genvar j = 0; j < 2; ++j) begin : g_extract
            localparam OFF = (i % 2) * 16 + j * 8;

            wire [7:0] raw_a = a_row[i/2][OFF +: 8];
            wire [7:0] raw_b = b_col[i/2][OFF +: 8];

            logic [4:0] raw_ea, raw_eb;
            logic [2:0] raw_ma, raw_mb;
            logic       raw_sa, raw_sb;
            logic [4:0] exp_max;

            always_comb begin
                if (is_bfloat) begin // TCU_BF8_ID (E5M2)
                    raw_ea  = 5'(raw_a[S_BF8-1 -: E_BF8]);
                    raw_eb  = 5'(raw_b[S_BF8-1 -: E_BF8]);
                    raw_ma  = {raw_a[1:0], 1'b0};
                    raw_mb  = {raw_b[1:0], 1'b0};
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                    exp_max = 5'h1F;
                end else begin       // TCU_FP8_ID (E4M3)
                    raw_ea  = 5'(raw_a[S_FP8-1 -: E_FP8]);
                    raw_eb  = 5'(raw_b[S_FP8-1 -: E_FP8]);
                    raw_ma  = raw_a[2:0];
                    raw_mb  = raw_b[2:0];
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                    exp_max = 5'h0F;
                end
            end

            // Generic Classifier Logic

            fedp_class_t cls_a;
            VX_tcu_tfr_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_a (
                .exp (raw_ea),
                .man (raw_ma),
                .max_exp (exp_max),
                .cls (cls_a)
            );

            fedp_class_t cls_b;
            VX_tcu_tfr_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_b (
                .exp (raw_eb),
                .man (raw_mb),
                .max_exp (exp_max),
                .cls (cls_b)
            );

            // Select normalized exponents (force subnormals to 1)
            assign ea_sel[j] = cls_a.is_sub ? 5'b1 : raw_ea;
            assign eb_sel[j] = cls_b.is_sub ? 5'b1 : raw_eb;

            // Select normalized mantissas (append implicit bit, force 0 if zero/subnormal)
            assign ma_sel[j] = {!cls_a.is_sub && !cls_a.is_zero, raw_a[2:0]};
            assign mb_sel[j] = {!cls_b.is_sub && !cls_b.is_zero, raw_b[2:0]};

            assign sign_sel[j] = raw_sa ^ raw_sb;
            assign zero_sel[j] = cls_a.is_zero | cls_b.is_zero;

            // Exception resolution
            wire nan_in = cls_a.is_nan | cls_b.is_nan;
            wire inf_z  = (cls_a.is_inf & cls_b.is_zero) | (cls_a.is_zero & cls_b.is_inf);
            wire inf_op = cls_a.is_inf | cls_b.is_inf;

            assign nan_sel[j] = nan_in | inf_z;
            assign inf_sel[j] = inf_op & ~inf_z;
        end

        // ------------------------------------------------------------------
        // 2b. Muxing (Bias only, rest is handled inline)
        // ------------------------------------------------------------------

        // Calculate Pre-Sums (Exponent A + Exponent B)

        wire [5:0] pre_sum_0, pre_sum_1;

        VX_ks_adder #(
            .N(6),
            .BYPASS(`FORCE_BUILTIN_ADDER(6))
        ) exp_add1_0 (
            .dataa(6'(ea_sel[0])),
            .datab(6'(eb_sel[0])),
            .cin(1'b0),
            .sum(pre_sum_0),
            `UNUSED_PIN(cout)
        );

        VX_ks_adder #(
            .N(6),
            .BYPASS(`FORCE_BUILTIN_ADDER(6))
        ) exp_add1_1 (
            .dataa(6'(ea_sel[1])),
            .datab(6'(eb_sel[1])),
            .cin(1'b0),
            .sum(pre_sum_1),
            `UNUSED_PIN(cout)
        );

        // Select max/min term using per-element validity.

        wire v0 = ~zero_sel[0] && lane_valid[0];
        wire v1 = ~zero_sel[1] && lane_valid[1];

        wire term0_is_max = (v0 & ~v1) || (pre_sum_0 >= pre_sum_1);
        wire diff_sign = term0_is_max;

        wire [5:0] max_pre_sum = term0_is_max ? pre_sum_0 : pre_sum_1;
        wire [5:0] min_pre_sum = term0_is_max ? pre_sum_1 : pre_sum_0;

        wire [7:0] bias_sel = is_bfloat ? BIAS_CONST_BF8 : BIAS_CONST_FP8;

        wire [EXP_W-1:0] final_exp;
        VX_ks_adder #(
            .N(EXP_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_final_add (
            .dataa(EXP_W'(max_pre_sum)),
            .datab(EXP_W'(bias_sel)),
            .cin(1'b0),
            .sum(final_exp),
            `UNUSED_PIN(cout)
        );

        // Exponent is meaningful only if lane has at least one valid contributing element.
        assign result_exp[i] = (v0 || v1) ? final_exp : '0;

        // ------------------------------------------------------------------
        // 2c. Mantissa Multiplication
        // ------------------------------------------------------------------

        wire [1:0][7:0] man_prod;

        for (genvar j = 0; j < 2; ++j) begin : g_mul
            VX_wallace_mul #(
                .N(4),
                .CPA_KS(!`FORCE_BUILTIN_ADDER(4*2))
            ) wtmul (
                .a(ma_sel[j]),
                .b(mb_sel[j]),
                .p(man_prod[j])
            );
        end

        // ------------------------------------------------------------------
        // 2d. Alignment & Reduction
        // ------------------------------------------------------------------

        wire [5:0] diff_abs;
        VX_ks_adder #(
            .N(6),
            .BYPASS(`FORCE_BUILTIN_ADDER(6))
        ) ks_diff (
            .dataa(max_pre_sum),
            .datab(~min_pre_sum),
            .cin(1'b1),
            .sum(diff_abs),
            `UNUSED_PIN(cout)
        );

        wire [7:0] man_prod0_v = man_prod[0] & {8{lane_valid[0]}};
        wire [7:0] man_prod1_v = man_prod[1] & {8{lane_valid[1]}};

        wire [22:0] sig_low  = {man_prod0_v, 15'b0};
        wire [22:0] sig_high = {man_prod1_v, 15'b0};

        // Prevent shift wrap when exponent diff >= 32:
        // diff_abs[5] indicates diff >= 32 (since diff_abs is 6-bit).
        // For such large diffs the smaller term is effectively zero.
        wire diff_ge_32 = diff_abs[5];
        wire [4:0] shamt = diff_abs[4:0];

        wire [22:0] aligned_sig_low  = diff_sign ? sig_low : (diff_ge_32 ? 23'b0 : (sig_low >> shamt));
        wire [22:0] aligned_sig_high = diff_sign ? (diff_ge_32 ? 23'b0 : (sig_high >> shamt)) : sig_high;

        // ------------------------------------------------------------------
        // 2e. Absolute Difference / Addition
        // ------------------------------------------------------------------
        wire [23:0] mag_0 = {1'b0, aligned_sig_low};
        wire [23:0] mag_1 = {1'b0, aligned_sig_high};
        wire mag_0_is_larger = (mag_0 > mag_1);

        wire [23:0] op_a = mag_0_is_larger ? mag_0 : mag_1;
        wire [23:0] op_b = mag_0_is_larger ? mag_1 : mag_0;

        wire do_sub = sign_sel[0] ^ sign_sel[1];

        wire [23:0] sig_add;
        VX_ks_adder #(
            .N(24),
            .BYPASS(`FORCE_BUILTIN_ADDER(24))
        ) sig_adder_f8 (
            .cin(do_sub),
            .dataa(op_a),
            .datab(do_sub ? ~op_b : op_b),
            .sum(sig_add),
            `UNUSED_PIN(cout)
        );

        // Force +0 on exact cancellation
        wire sig_sign_raw = mag_0_is_larger ? sign_sel[0] : sign_sel[1];
        wire sig_sign = (sig_add == 24'b0) ? 1'b0 : sig_sign_raw;
        assign result_sig[i] = {sig_sign, sig_add};

        // ------------------------------------------------------------------
        // 2f. Exception Merging (Merge 2 sub-products per lane)
        // ------------------------------------------------------------------

        // Check for +Inf + -Inf (Generates NaN)
        wire pos_inf_0 = inf_sel[0] && ~sign_sel[0] && lane_valid[0];
        wire neg_inf_0 = inf_sel[0] &&  sign_sel[0] && lane_valid[0];
        wire pos_inf_1 = inf_sel[1] && ~sign_sel[1] && lane_valid[1];
        wire neg_inf_1 = inf_sel[1] &&  sign_sel[1] && lane_valid[1];

        wire add_nan = (pos_inf_0 && neg_inf_1) || (neg_inf_0 && pos_inf_1);

        wire any_nan = (|(nan_sel & lane_valid)) || add_nan;
        wire any_inf = (|(inf_sel & lane_valid)) && ~any_nan;

        // Result Sign:
        // If one is Inf, take its sign.
        // If both are Inf (same sign), take that sign.
        // If neither is Inf, we rely on the arithmetic result sign (computed above).
        wire final_sign_inf = inf_sel[0] ? sign_sel[0] : sign_sel[1];

        assign exceptions[i].is_nan = any_nan;
        assign exceptions[i].is_inf = any_inf;
        assign exceptions[i].sign   = any_inf ? final_sign_inf : sig_sign;

    end

endmodule
