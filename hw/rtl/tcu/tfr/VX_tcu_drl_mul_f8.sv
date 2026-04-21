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

    // ======================================================================
    // 2. Main Loop (Per TCK Lane)
    // ======================================================================

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire is_bfloat = tcu_fmt_is_bfloat(fmt_f);

        // Per-element valid bits (2 elements -> 1 lane)
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

            wire is_ea_zero = (raw_ea == 0);
            wire is_eb_zero = (raw_eb == 0);

            // Select normalized exponents
            assign ea_sel[j] = is_ea_zero ? 5'b1 : raw_ea;
            assign eb_sel[j] = is_eb_zero ? 5'b1 : raw_eb;

            // Select normalized mantissas
            assign ma_sel[j] = {~is_ea_zero, raw_ma};
            assign mb_sel[j] = {~is_eb_zero, raw_mb};
            `UNUSED_VAR (cls_a.is_sub)
            `UNUSED_VAR (cls_b.is_sub)

            assign sign_sel[j] = raw_sa ^ raw_sb;
            assign zero_sel[j] = cls_a.is_zero | cls_b.is_zero;

            // Exception resolution
            wire a_is_inf = is_bfloat ? cls_a.is_inf : 1'b0;
            wire b_is_inf = is_bfloat ? cls_b.is_inf : 1'b0;
            wire a_is_nan = is_bfloat ? cls_a.is_nan : (raw_ea == 5'h0F) && (raw_ma == 3'b111);
            wire b_is_nan = is_bfloat ? cls_b.is_nan : (raw_eb == 5'h0F) && (raw_mb == 3'b111);

            wire nan_in = a_is_nan | b_is_nan;
            wire inf_z  = (a_is_inf & cls_b.is_zero) | (cls_a.is_zero & b_is_inf);
            wire inf_op = a_is_inf | b_is_inf;

            assign nan_sel[j] = nan_in | inf_z;
            assign inf_sel[j] = inf_op & ~inf_z;
        end

        // ------------------------------------------------------------------
        // 2b. Speculative Exponent Difference
        // ------------------------------------------------------------------
        // Compute both diffs in parallel — use sign bit for comparison
        // instead of serial: compare → mux → subtract

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

        wire v0 = ~zero_sel[0] && lane_valid[0];
        wire v1 = ~zero_sel[1] && lane_valid[1];

        // Speculative diffs: computed in parallel, not after compare
        wire [6:0] diff_0_minus_1 = {1'b0, pre_sum_0} - {1'b0, pre_sum_1};
        wire [6:0] diff_1_minus_0 = {1'b0, pre_sum_1} - {1'b0, pre_sum_0};
        `UNUSED_VAR (diff_1_minus_0[6])

        // Comparison from sign bit — no separate comparator needed
        wire term0_ge_term1 = ~diff_0_minus_1[6];
        wire term0_is_max = (v0 & ~v1) || (v1 & term0_ge_term1);
        wire diff_sign = term0_is_max;

        // Select absolute difference (just a mux, not a new subtraction)
        wire [5:0] diff_abs = term0_is_max ? diff_0_minus_1[5:0] : diff_1_minus_0[5:0];

        // Max exponent sum (for output, no bias added)
        wire [5:0] max_pre_sum = term0_is_max ? pre_sum_0 : pre_sum_1;

        // Per-lane bias addition
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
        // 2d. Pre-Sort & Alignment
        // ------------------------------------------------------------------

        wire [7:0] man_prod0_v = man_prod[0] & {8{lane_valid[0]}};
        wire [7:0] man_prod1_v = man_prod[1] & {8{lane_valid[1]}};

        // Sort products by exponent magnitude (8-bit mux, not 24-bit)
        wire [7:0] prod_max = diff_sign ? man_prod0_v : man_prod1_v;
        wire [7:0] prod_min = diff_sign ? man_prod1_v : man_prod0_v;
        wire       sign_max = diff_sign ? sign_sel[0] : sign_sel[1];
        wire       sign_min = diff_sign ? sign_sel[1] : sign_sel[0];

        // Pad to 24 bits; barrel shift only the min term
        wire [23:0] sig_max = {prod_max, 16'b0};
        wire diff_ge_32 = diff_abs[5];
        wire [4:0] shamt = diff_abs[4:0];
        wire [23:0] sig_min_shifted = diff_ge_32 ? 24'b0 : ({prod_min, 16'b0} >> shamt);

        // ------------------------------------------------------------------
        // 2e. Add/Sub Reduction
        // ------------------------------------------------------------------

        wire [24:0] sig_max_ext = {1'b0, sig_max};
        wire [24:0] sig_min_ext = {1'b0, sig_min_shifted};

        wire do_sub = sign_sel[0] ^ sign_sel[1];

        wire [24:0] add_raw;
        VX_ks_adder #(
            .N(25),
            .BYPASS(`FORCE_BUILTIN_ADDER(25))
        ) add_adder (
            .dataa(sig_max_ext),
            .datab(sig_min_ext),
            .cin(1'b0),
            .sum(add_raw),
            `UNUSED_PIN(cout)
        );

        wire [24:0] sub_raw;
        VX_ks_adder #(
            .N(25),
            .BYPASS(`FORCE_BUILTIN_ADDER(25))
        ) sub_adder (
            .dataa(sig_max_ext),
            .datab(~sig_min_ext),
            .cin(1'b1),
            .sum(sub_raw),
            `UNUSED_PIN(cout)
        );

        wire sub_neg = sub_raw[24];
        wire [24:0] sub_abs = sub_neg ? -sub_raw : sub_raw;

        wire [24:0] sig_add_raw = do_sub ? sub_abs : add_raw;

        // Scaling by 1 to avoid renormalization
        wire [23:0] sig_add = sig_add_raw[24:1];
        `UNUSED_VAR (sig_add_raw[0])

        // Exact cancellation detection
        wire pre_sum_eq = (pre_sum_0 == pre_sum_1);
        wire mag_is_equal = pre_sum_eq && (man_prod0_v == man_prod1_v);
        wire is_zero_out = do_sub && mag_is_equal;

        // Force +0 on exact cancellation
        wire sig_sign_raw = sub_neg ? sign_min : sign_max;
        wire sig_sign = is_zero_out ? 1'b0 : sig_sign_raw;

        assign result_sig[i] = {sig_sign, sig_add};
        assign result_exp[i] = ((v0 || v1) && !is_zero_out) ? final_exp : '0;

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

        wire final_sign_inf = inf_sel[0] ? sign_sel[0] : sign_sel[1];

        assign exceptions[i].is_nan = any_nan;
        assign exceptions[i].is_inf = any_inf;
        assign exceptions[i].sign   = any_inf ? final_sign_inf : sig_sign;

    `ifdef DBG_TRACE_TCU
        always_ff @(posedge clk) begin
            if (valid_in && g_lane[i].lane_valid != 0) begin
                `TRACE(4, ("%t: %s FEDP-REDUCE(%0d) lane=%0d: ", $time, INSTANCE_ID, req_id, i));
                `TRACE(4, ("max=(%0d, 0x%0h, %0d), ", sign_max, sig_max, max_pre_sum));
                `TRACE(4, ("min=(%0d, 0x%0h) ", sign_min, sig_min_shifted));
                `TRACE(4, ("-> s=%0d, P=0x%0h, E=%0d\n", sig_sign, sig_add, final_exp));
            end
        end
    `endif

    end

endmodule
