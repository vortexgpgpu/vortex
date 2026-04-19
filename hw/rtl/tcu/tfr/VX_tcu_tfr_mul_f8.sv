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
    output logic [TCK-1:0][W-1:0]     result_sig,
    output logic [TCK-1:0][EXP_W-1:0] result_exp,
    output fedp_excep_t [TCK-1:0]     exceptions
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})
    `UNUSED_VAR (vld_mask)
    `UNUSED_VAR ({WA})

    // ======================================================================
    // 1. Constants & Parameters
    // ======================================================================

    // Narrowed reduction: 8-bit products need at most 8-bit alignment room
    localparam F8_DATA_W = 16; // 8-bit product + 8-bit alignment
    localparam F8_ADD_W  = F8_DATA_W + 1; // 17-bit for addition with carry

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam S_FP8 = VX_tcu_pkg::sign_pos(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1;

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam S_BF8 = VX_tcu_pkg::sign_pos(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1;

    `UNUSED_PARAM (B_FP8)
    `UNUSED_PARAM (B_BF8)

    wire is_bfloat = tcu_fmt_is_bfloat(fmt_f);

    // ======================================================================
    // 2. Main Loop (Per TCK Lane)
    // ======================================================================

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

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

            // Select normalized exponents
            assign ea_sel[j] = (raw_ea == 0) ? 5'b1 : raw_ea;
            assign eb_sel[j] = (raw_eb == 0) ? 5'b1 : raw_eb;

            // Select normalized mantissas
            assign ma_sel[j] = {(raw_ea != 0), raw_ma};
            assign mb_sel[j] = {(raw_eb != 0), raw_mb};
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

        // Comparison from sign bit — no separate comparator needed
        wire term0_ge_term1 = ~diff_0_minus_1[6];
        wire term0_is_max = (v0 & ~v1) || (v1 & term0_ge_term1);
        wire diff_sign = term0_is_max;

        // Select absolute difference (just a mux, not a new subtraction)
        wire [5:0] diff_abs = term0_is_max ? diff_0_minus_1[5:0] : diff_1_minus_0[5:0];

        // Max exponent sum (for output, no bias added)
        wire [5:0] max_pre_sum = term0_is_max ? pre_sum_0 : pre_sum_1;

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
        // 2d. Narrowed Alignment & Reduction
        // ------------------------------------------------------------------
        // 8-bit products: shifts >= 8 flush to zero (sticky only)
        // Data path narrowed from 24-bit to 16-bit
        // Adders narrowed from 25-bit to 17-bit

        wire [7:0] man_prod0_v = man_prod[0] & {8{lane_valid[0]}};
        wire [7:0] man_prod1_v = man_prod[1] & {8{lane_valid[1]}};

        // Pad to F8_DATA_W (16 bits)
        wire [F8_DATA_W-1:0] sig_low  = {man_prod0_v, 8'b0};
        wire [F8_DATA_W-1:0] sig_high = {man_prod1_v, 8'b0};

        // Narrowed barrel shifter: 3-bit control, flush if >= 8
        wire flush = |diff_abs[5:3]; // diff_abs >= 8
        wire [2:0] shamt = diff_abs[2:0];
        wire [F8_DATA_W-1:0] aligned_sig_low  = diff_sign ? sig_low  : (flush ? '0 : (sig_low  >> shamt));
        wire [F8_DATA_W-1:0] aligned_sig_high = diff_sign ? (flush ? '0 : (sig_high >> shamt)) : sig_high;

        // ------------------------------------------------------------------
        // 2e. Narrowed Absolute Difference / Addition (17-bit)
        // ------------------------------------------------------------------

        wire [F8_ADD_W-1:0] sig_low_ext  = {1'b0, aligned_sig_low};
        wire [F8_ADD_W-1:0] sig_high_ext = {1'b0, aligned_sig_high};

        wire do_sub = sign_sel[0] ^ sign_sel[1];

        wire [F8_ADD_W-1:0] sub_lh = sig_low_ext - sig_high_ext;
        wire [F8_ADD_W-1:0] sub_hl = sig_high_ext - sig_low_ext;
        wire [F8_ADD_W-1:0] add_raw = sig_low_ext + sig_high_ext;

        wire sub_neg = sub_lh[F8_ADD_W-1];
        wire [F8_ADD_W-1:0] sub_abs = sub_neg ? sub_hl : sub_lh;

        wire mag_0_is_larger = ~sub_neg;
        wire [F8_ADD_W-1:0] sig_add_raw = do_sub ? sub_abs : add_raw;

        // Scaling by 1 to avoid renormalization
        wire [F8_DATA_W-1:0] sig_add = sig_add_raw[F8_ADD_W-1:1];
        `UNUSED_VAR (sig_add_raw[0])

        // Exact cancellation detection
        wire pre_sum_eq = (pre_sum_0 == pre_sum_1);
        wire mag_is_equal = pre_sum_eq && (man_prod0_v == man_prod1_v);
        wire is_zero_out = do_sub && mag_is_equal;

        // Force +0 on exact cancellation
        wire sig_sign_raw = mag_0_is_larger ? sign_sel[0] : sign_sel[1];
        wire sig_sign = is_zero_out ? 1'b0 : sig_sign_raw;

        // Output: pad narrowed result to W bits
        assign result_sig[i] = {sig_sign, sig_add, {(W - 1 - F8_DATA_W){1'b0}}};
        // Output: raw exponent (no bias), zero-extended to EXP_W
        assign result_exp[i] = ((v0 || v1) && !is_zero_out) ? EXP_W'(max_pre_sum) : '0;

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
                `TRACE(4, ("t1=(%0d, 0x%0h, %0d), ", sign_sel[0], sig_low, pre_sum_0));
                `TRACE(4, ("t2=(%0d, 0x%0h, %0d) ", sign_sel[1], sig_high, pre_sum_1));
                `TRACE(4, ("| aln_t1=0x%0h, aln_t2=0x%0h ", aligned_sig_low, aligned_sig_high));
                `TRACE(4, ("-> s=%0d, P=0x%0h, E=%0d\n", sig_sign, sig_add, result_exp[i]));
            end
        end
    `endif

    end

endmodule
