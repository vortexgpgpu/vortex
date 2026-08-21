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

module VX_tcu_tfr_mul_f4 import VX_tcu_pkg::*;
#(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10,
    parameter USE_DSP = 0,  // map mantissa multipliers onto DSP48 slices
    parameter PROD_REG = 0  // product/flag register stages (multiply-stage seam)
) (
    input wire                      clk,
    input wire                      enable,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [3:0]                fmt_f,

    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,
`ifdef VX_CFG_TCU_MX_ENABLE
    input wire [7:0]                sf_a,
    input wire [7:0]                sf_b,
`endif

    // result_exp/exceptions are pre-seam (classify cycle); result_sig and
    // sig_zero are post-seam (PROD_REG cycles later). sig_zero flags exact
    // cancellation of the term sum: the joined exponent must be zeroed so
    // the lane is excluded from the max-exponent search.
    output logic [TCK-1:0][24:0]      result_sig,
    output logic [TCK-1:0][EXP_W-1:0] result_exp,
    output fedp_excep_t [TCK-1:0]     exceptions,
    output logic [TCK-1:0]            sig_zero
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_SPARAM (W)
    `UNUSED_VAR ({clk, enable, req_id, valid_in, fmt_f})

`ifdef VX_CFG_TCU_MX_ENABLE
`ifdef VX_CFG_TCU_FP4_ENABLE

    localparam [3:0] RZR4_SPECIAL_MAG_X2 = 4'd10;

    function automatic [3:0] rzr4_mag_x2(input logic [3:0] raw);
        case (raw)
            4'h0:   return RZR4_SPECIAL_MAG_X2;
            4'h1,
            4'h9:   return 4'd1;
            4'h2,
            4'ha:   return 4'd2;
            4'h3,
            4'hb:   return 4'd3;
            4'h4,
            4'hc:   return 4'd4;
            4'h5,
            4'hd:   return 4'd6;
            4'h6,
            4'he:   return 4'd8;
            4'h7,
            4'hf:   return 4'd12;
            default:return 4'd0;
        endcase
    endfunction
    // Post-seam format select for the significand path.
    wire [3:0] fmt_f_r;
    VX_pipe_register #(
        .DATAW (4),
        .DEPTH (PROD_REG)
    ) pipe_fmt (
        .clk      (clk),
        .reset    (1'b0),
        .enable   (enable),
        .data_in  (fmt_f),
        .data_out (fmt_f_r)
    );

`ifdef VX_CFG_TCU_MXFP4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_mxfp4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_mxfp4;
    fedp_excep_t [TCK-1:0]    exceptions_mxfp4;
    wire [TCK-1:0]            sig_zero_mxfp4;

    localparam F32_BIAS_MXFP4  = 127;
    localparam S_FP32_MXFP4    = 23;
    localparam S_SUPER_MXFP4   = 22;
    localparam BIAS_BASE_MXFP4 = F32_BIAS_MXFP4 + 2*(S_FP32_MXFP4 - S_SUPER_MXFP4) - W + WA - 1 + 128;

    localparam SIG_SHIFT_MXFP4 = 11;

    localparam F4_EXP_BIAS_MXFP4   = 1;
    localparam EXP_TERM_W_MXFP4    = 6;
    localparam EXP_TERM_BIAS_MXFP4 = 1 << (EXP_TERM_W_MXFP4 - 1);
    localparam EXP_COMP_MXFP4      = -(2 * F4_EXP_BIAS_MXFP4 + 10);
    localparam [EXP_TERM_W_MXFP4-1:0] EXP_ADJ_MXFP4 = EXP_TERM_W_MXFP4'(EXP_TERM_BIAS_MXFP4 + EXP_COMP_MXFP4);
    localparam [EXP_W-1:0] EXP_BASE_BIASED_MXFP4 = EXP_W'(BIAS_BASE_MXFP4 + EXP_COMP_MXFP4);

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_mxfp4
        localparam K_WORD = i / 2;

        wire [3:0][23:0] term_mag_shifted;
        wire [3:0][EXP_TERM_W_MXFP4-1:0] term_exp_biased;
        wire [3:0] term_valid;
        wire [3:0] term_sign;
        wire [3:0][1:0] a_man, b_man;
        wire [3:0][3:0] f4_man_prod;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;

            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            wire a_zero = ~|raw_a[2:0];
            wire b_zero = ~|raw_b[2:0];
            assign term_valid[j] = lane_valid && !a_zero && !b_zero;
            assign term_sign[j]  = raw_a[3] ^ raw_b[3];

            assign a_man[j] = ~|raw_a[2:1] ? 2'b01 : {1'b1, raw_a[0]};
            assign b_man[j] = ~|raw_b[2:1] ? 2'b01 : {1'b1, raw_b[0]};

            wire [1:0] a_exp, b_exp;
            assign a_exp[0] = raw_a[2] & ~raw_a[1];
            assign a_exp[1] = raw_a[2] & raw_a[1];
            assign b_exp[0] = raw_b[2] & ~raw_b[1];
            assign b_exp[1] = raw_b[2] & raw_b[1];

            wire signed [9:0] sf_exp_a = $signed({1'b0, sf_a}) - 10'sd127;
            wire signed [9:0] sf_exp_b = $signed({1'b0, sf_b}) - 10'sd127;
            wire signed [5:0] exp_biased_raw = 6'(10'(EXP_ADJ_MXFP4)
                                                + sf_exp_a
                                                + sf_exp_b
                                                + 10'(a_exp)
                                                + 10'(b_exp));

            assign term_exp_biased[j] = term_valid[j] ? exp_biased_raw : '0;
        end

        // Pack the four 2x2 mantissa products into two DSP48s (two per DSP);
        // PROD_REG lands the products in the DSP48 PREG.
        VX_tcu_tfr_wmul #(
            .N       (2),
            .LANES   (2),
            .USE_DSP (USE_DSP),
            .OUT_REG (PROD_REG)
        ) f4m01 (
            .clk    (clk),
            .enable (enable),
            .a (a_man[1:0]),
            .b (b_man[1:0]),
            .p (f4_man_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N       (2),
            .LANES   (2),
            .USE_DSP (USE_DSP),
            .OUT_REG (PROD_REG)
        ) f4m23 (
            .clk    (clk),
            .enable (enable),
            .a (a_man[3:2]),
            .b (b_man[3:2]),
            .p (f4_man_prod[3:2])
        );

        wire [EXP_TERM_W_MXFP4-1:0] max_exp_01 = (term_exp_biased[0] >= term_exp_biased[1]) ? term_exp_biased[0] : term_exp_biased[1];
        wire [EXP_TERM_W_MXFP4-1:0] max_exp_23 = (term_exp_biased[2] >= term_exp_biased[3]) ? term_exp_biased[2] : term_exp_biased[3];
        wire [EXP_TERM_W_MXFP4-1:0] max_exp_biased = (max_exp_01 >= max_exp_23) ? max_exp_01 : max_exp_23;

        // Term alignment controls depend only on exponents and signs:
        // pre-seam compute, post-seam use.
        wire [3:0] term_valid_r, term_sign_r;
        wire [3:0][EXP_TERM_W_MXFP4-1:0] shift_amt_r;
        wire [3:0][EXP_TERM_W_MXFP4-1:0] shift_amt_w;
        VX_pipe_register #(
            .DATAW (4 + 4 + 4 * EXP_TERM_W_MXFP4),
            .DEPTH (PROD_REG)
        ) pipe_ctrl (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  ({term_valid,   term_sign,   shift_amt_w}),
            .data_out ({term_valid_r, term_sign_r, shift_amt_r})
        );

        wire [3:0][26:0] term_signed;
        for (genvar j = 0; j < 4; ++j) begin : g_align
            VX_ks_adder #(
                .N(EXP_TERM_W_MXFP4),
                .BYPASS(`FORCE_BUILTIN_ADDER(EXP_TERM_W_MXFP4))
            ) shift_ksa (
                .dataa(max_exp_biased),
                .datab(~term_exp_biased[j]),
                .cin(1'b1),
                .sum(shift_amt_w[j]),
                `UNUSED_PIN(cout)
            );

            assign term_mag_shifted[j] = term_valid_r[j] ? (24'(f4_man_prod[j]) << SIG_SHIFT_MXFP4) : 24'd0;
            wire [23:0] aligned_mag = (shift_amt_r[j] >= EXP_TERM_W_MXFP4'(24)) ? 24'd0 : (term_mag_shifted[j] >> shift_amt_r[j][4:0]);
            wire [26:0] aligned_ext = {3'b0, aligned_mag};

            wire [26:0] neg_term;
            VX_ks_adder #(
                .N(27),
                .BYPASS(`FORCE_BUILTIN_ADDER(27))
            ) term_neg_ksa (
                .dataa(~aligned_ext),
                .datab(27'd0),
                .cin(1'b1),
                .sum(neg_term),
                `UNUSED_PIN(cout)
            );

            assign term_signed[j] = term_sign_r[j] ? neg_term : aligned_ext;
        end

        wire [26:0] sum_vec, carry_vec;
        VX_csa_tree #(
            .N(4),
            .W(27),
            .S(27)
        ) term_csa (
            .operands (term_signed),
            .sum      (sum_vec),
            .carry    (carry_vec)
        );

        wire [26:0] signed_sum;
        VX_ks_adder #(
            .N(27),
            .BYPASS(`FORCE_BUILTIN_ADDER(27))
        ) sum_ksa (
            .dataa(sum_vec),
            .datab(carry_vec),
            .cin(1'b0),
            .sum(signed_sum),
            `UNUSED_PIN(cout)
        );

        wire sum_sign = signed_sum[26];
        wire [25:0] neg_sum_raw;
        VX_ks_adder #(
            .N(26),
            .BYPASS(`FORCE_BUILTIN_ADDER(26))
        ) sum_neg_ksa (
            .dataa(~signed_sum[25:0]),
            .datab(26'd0),
            .cin(1'b1),
            .sum(neg_sum_raw),
            `UNUSED_PIN(cout)
        );

        wire [25:0] abs_sum = sum_sign ? neg_sum_raw : signed_sum[25:0];
        wire is_zero_out = ~|abs_sum;

        assign result_sig_mxfp4[i] = {sum_sign & ~is_zero_out, abs_sum[23:0]};
        assign sig_zero_mxfp4[i]   = is_zero_out;
        assign result_exp_mxfp4[i] = EXP_W'(max_exp_biased) + EXP_W'(EXP_BASE_BIASED_MXFP4);

        // The sign field is only consumed for infinity lanes downstream.
        assign exceptions_mxfp4[i].is_nan = 1'b0;
        assign exceptions_mxfp4[i].is_inf = 1'b0;
        assign exceptions_mxfp4[i].sign   = 1'b0;
    end
`endif  // VX_CFG_TCU_MXFP4_ENABLE

`ifdef VX_CFG_TCU_NVFP4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_nvfp4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_nvfp4;
    fedp_excep_t [TCK-1:0]    exceptions_nvfp4;
    wire [TCK-1:0]            sig_zero_nvfp4;

    localparam F32_BIAS  = 127;
    localparam S_FP32    = 23;
    localparam S_SUPER   = 22;
    localparam BIAS_BASE = F32_BIAS + 2*(S_FP32 - S_SUPER) - W + WA - 1 + 128;

    localparam SIG_SHIFT = 11;

    localparam F4_EXP_BIAS     = 1;
    localparam SF_EXP_BIAS     = 7;
    localparam SF_MAN_BITS     = 3;
    localparam EXP_TERM_W      = 6;
    localparam EXP_TERM_BIAS   = 1 << (EXP_TERM_W - 1);
    // fp4 = man * 2^(exp - 1), e4m3 scale = man * 2^(exp - 7 - 3).
    localparam EXP_COMP_NVFP4  = -(2 * F4_EXP_BIAS + 2 * (SF_EXP_BIAS + SF_MAN_BITS));
    localparam [5:0] EXP_ADJ_NVFP4 = 6'(EXP_TERM_BIAS + EXP_COMP_NVFP4);
    localparam [EXP_W-1:0] EXP_BASE_BIASED = EXP_W'(BIAS_BASE + EXP_COMP_NVFP4);

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_nvfp4
        localparam K_WORD = i / 2;
        `UNUSED_VAR ({sf_a[7], sf_b[7]})

        // e4m3 scale factor mantissa mul
        wire [3:0] sf_man_a = {1'b1, sf_a[2:0]};
        wire [3:0] sf_man_b = {1'b1, sf_b[2:0]};
        wire [3:0] sf_exp_a = sf_a[6:3];
        wire [3:0] sf_exp_b = sf_b[6:3];

        wire [7:0] sf_man_prod;
        VX_tcu_tfr_wmul #(
            .N(4),
            .USE_DSP(USE_DSP)
        ) sf_wtmul (
            .clk    (clk),
            .enable (enable),
            .a(sf_man_a),
            .b(sf_man_b),
            .p(sf_man_prod)
        );

        wire [3:0][23:0] term_mag_shifted;
        wire [3:0][5:0]  term_exp_biased;
        wire [3:0]       term_valid;
        wire [3:0]       term_sign;
        wire [3:0][1:0]  a_man, b_man;
        wire [3:0][3:0]  f4_man_prod;
        wire [3:0][11:0] term_man_prod;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;

            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            wire a_zero = ~|raw_a[2:0];
            wire b_zero = ~|raw_b[2:0];
            assign term_valid[j] = lane_valid && !a_zero && !b_zero;
            assign term_sign[j]  = raw_a[3] ^ raw_b[3];

            assign a_man[j] = ~|raw_a[2:1] ? 2'b01 : {1'b1, raw_a[0]};
            assign b_man[j] = ~|raw_b[2:1] ? 2'b01 : {1'b1, raw_b[0]};

            wire [1:0] a_exp, b_exp;
            assign a_exp[0] = raw_a[2] & ~raw_a[1];
            assign a_exp[1] = raw_a[2] & raw_a[1];
            assign b_exp[0] = raw_b[2] & ~raw_b[1];
            assign b_exp[1] = raw_b[2] & raw_b[1];

            wire [5:0] exp_sum_vec, exp_carry_vec;
            VX_csa_tree #(
                .N(5),
                .W(6),
                .S(6)
            ) exp_csa (
                .operands ({EXP_ADJ_NVFP4, 6'(sf_exp_a), 6'(sf_exp_b), 6'(a_exp), 6'(b_exp)}),
                .sum      (exp_sum_vec),
                .carry    (exp_carry_vec)
            );

            wire [5:0] exp_biased_raw;
            VX_ks_adder #(
                .N(6),
                .BYPASS(`FORCE_BUILTIN_ADDER(6))
            ) exp_ksa (
                .dataa(exp_sum_vec),
                .datab(exp_carry_vec),
                .cin(1'b0),
                .sum(exp_biased_raw),
                `UNUSED_PIN(cout)
            );

            assign term_exp_biased[j] = term_valid[j] ? exp_biased_raw : 6'd0;
        end

        // Pack the four 2x2 mantissa products into two DSP48s.
        VX_tcu_tfr_wmul #(
            .N       (2),
            .LANES   (2),
            .USE_DSP (USE_DSP)
        ) f4m01 (
            .clk    (clk),
            .enable (enable),
            .a (a_man[1:0]),
            .b (b_man[1:0]),
            .p (f4_man_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N       (2),
            .LANES   (2),
            .USE_DSP (USE_DSP)
        ) f4m23 (
            .clk    (clk),
            .enable (enable),
            .a (a_man[3:2]),
            .b (b_man[3:2]),
            .p (f4_man_prod[3:2])
        );

        // Each term scales its 2x2 product by the SHARED per-lane scale-factor
        // mantissa (sf_man_prod) -> shared-operand packing, two terms per DSP48.
        // PROD_REG lands the scaled products in the DSP48 PREG.
        VX_tcu_tfr_wmul #(
            .N        (4),
            .M        (8),
            .LANES    (2),
            .SHARED_B (1),
            .USE_DSP  (USE_DSP),
            .OUT_REG  (PROD_REG)
        ) tm01 (
            .clk    (clk),
            .enable (enable),
            .a (f4_man_prod[1:0]),
            .b ({8'b0, sf_man_prod}),
            .p (term_man_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N        (4),
            .M        (8),
            .LANES    (2),
            .SHARED_B (1),
            .USE_DSP  (USE_DSP),
            .OUT_REG  (PROD_REG)
        ) tm23 (
            .clk    (clk),
            .enable (enable),
            .a (f4_man_prod[3:2]),
            .b ({8'b0, sf_man_prod}),
            .p (term_man_prod[3:2])
        );

        wire [5:0] max_exp_01 = (term_exp_biased[0] >= term_exp_biased[1]) ? term_exp_biased[0] : term_exp_biased[1];
        wire [5:0] max_exp_23 = (term_exp_biased[2] >= term_exp_biased[3]) ? term_exp_biased[2] : term_exp_biased[3];
        wire [5:0] max_exp_biased = (max_exp_01 >= max_exp_23) ? max_exp_01 : max_exp_23;

        // Term alignment controls depend only on exponents and signs:
        // pre-seam compute, post-seam use.
        wire [3:0] term_valid_r, term_sign_r;
        wire [3:0][5:0] shift_amt_r;
        wire [3:0][5:0] shift_amt_w;
        VX_pipe_register #(
            .DATAW (4 + 4 + 4 * 6),
            .DEPTH (PROD_REG)
        ) pipe_ctrl (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  ({term_valid,   term_sign,   shift_amt_w}),
            .data_out ({term_valid_r, term_sign_r, shift_amt_r})
        );

        wire [3:0][26:0] term_signed;
        for (genvar j = 0; j < 4; ++j) begin : g_align
            VX_ks_adder #(
                .N(6),
                .BYPASS(`FORCE_BUILTIN_ADDER(6))
            ) shift_ksa (
                .dataa(max_exp_biased),
                .datab(~term_exp_biased[j]),
                .cin(1'b1),
                .sum(shift_amt_w[j]),
                `UNUSED_PIN(cout)
            );

            assign term_mag_shifted[j] = term_valid_r[j] ? (24'(term_man_prod[j][10:0]) << SIG_SHIFT) : 24'd0;
            wire [23:0] aligned_mag = (shift_amt_r[j] >= 6'd24) ? 24'd0 : (term_mag_shifted[j] >> shift_amt_r[j][4:0]);
            wire [26:0] aligned_ext = {3'b0, aligned_mag};

            wire [26:0] neg_term;
            VX_ks_adder #(
                .N(27),
                .BYPASS(`FORCE_BUILTIN_ADDER(27))
            ) term_neg_ksa (
                .dataa(~aligned_ext),
                .datab(27'd0),
                .cin(1'b1),
                .sum(neg_term),
                `UNUSED_PIN(cout)
            );

            assign term_signed[j] = term_sign_r[j] ? neg_term : aligned_ext;
        end

        wire [26:0] sum_vec, carry_vec;
        VX_csa_tree #(
            .N(4),
            .W(27),
            .S(27)
        ) term_csa (
            .operands (term_signed),
            .sum      (sum_vec),
            .carry    (carry_vec)
        );

        wire [26:0] signed_sum;
        VX_ks_adder #(
            .N(27),
            .BYPASS(`FORCE_BUILTIN_ADDER(27))
        ) sum_ksa (
            .dataa(sum_vec),
            .datab(carry_vec),
            .cin(1'b0),
            .sum(signed_sum),
            `UNUSED_PIN(cout)
        );

        wire sum_sign = signed_sum[26];
        wire [25:0] neg_sum_raw;
        VX_ks_adder #(
            .N(26),
            .BYPASS(`FORCE_BUILTIN_ADDER(26))
        ) sum_neg_ksa (
            .dataa(~signed_sum[25:0]),
            .datab(26'd0),
            .cin(1'b1),
            .sum(neg_sum_raw),
            `UNUSED_PIN(cout)
        );

        wire [25:0] abs_sum = sum_sign ? neg_sum_raw : signed_sum[25:0];
        wire is_zero_out = ~|abs_sum;

        assign result_sig_nvfp4[i] = {sum_sign & ~is_zero_out, abs_sum[23:0]};
        assign sig_zero_nvfp4[i]   = is_zero_out;
        assign result_exp_nvfp4[i] = EXP_W'(max_exp_biased) + EXP_W'(EXP_BASE_BIASED);

        // The sign field is only consumed for infinity lanes downstream.
        assign exceptions_nvfp4[i].is_nan = 1'b0;
        assign exceptions_nvfp4[i].is_inf = 1'b0;
        assign exceptions_nvfp4[i].sign   = 1'b0;
    end
`endif  // VX_CFG_TCU_NVFP4_ENABLE

`ifdef VX_CFG_TCU_RZR4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_rzr4;
    wire [TCK-1:0]            sig_zero_rzr4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_rzr4;
    fedp_excep_t [TCK-1:0]    exceptions_rzr4;

    localparam F32_BIAS_RZR4  = 127;
    localparam S_FP32_RZR4    = 23;
    localparam S_SUPER_RZR4   = 22;
    localparam BIAS_BASE_RZR4 = F32_BIAS_RZR4 + 2*(S_FP32_RZR4 - S_SUPER_RZR4) - W + WA - 1 + 128;
    localparam SF_EXP_BIAS_RZR4 = 7;
    localparam SF_MAN_BITS_RZR4 = 3;
    localparam EXP_TERM_W_RZR4 = 6;
    localparam EXP_TERM_BIAS_RZR4 = 1 << (EXP_TERM_W_RZR4 - 1);
    localparam EXP_COMP_RZR4 = -(2 + 2 * (SF_EXP_BIAS_RZR4 + SF_MAN_BITS_RZR4));
    localparam [EXP_TERM_W_RZR4-1:0] EXP_ADJ_RZR4 =
        EXP_TERM_W_RZR4'(EXP_TERM_BIAS_RZR4 + EXP_COMP_RZR4 + 4);
    localparam [EXP_W-1:0] EXP_BASE_BIASED_RZR4 =
        EXP_W'(BIAS_BASE_RZR4 + EXP_COMP_RZR4);
    localparam SIG_SHIFT_RZR4 = 7;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_rzr4
        localparam K_WORD = i / 2;

        wire [3:0] sf_exp_raw_a = sf_a[6:3];
        wire [3:0] sf_exp_raw_b = sf_b[6:3];
        wire [3:0] sf_exp_a = (sf_exp_raw_a == 0) ? 4'd1 : sf_exp_raw_a;
        wire [3:0] sf_exp_b = (sf_exp_raw_b == 0) ? 4'd1 : sf_exp_raw_b;
        wire [3:0] sf_man_a = (sf_exp_raw_a == 0) ? {1'b0, sf_a[2:0]} : {1'b1, sf_a[2:0]};
        wire [3:0] sf_man_b = (sf_exp_raw_b == 0) ? {1'b0, sf_b[2:0]} : {1'b1, sf_b[2:0]};
        wire scale_valid = (|sf_man_a) && (|sf_man_b);

        wire [7:0] sf_man_prod;
        VX_tcu_tfr_wmul #(
            .N(4),
            .USE_DSP(USE_DSP)
        ) sf_wtmul (
            .clk(clk),
            .enable(enable),
            .a(sf_man_a),
            .b(sf_man_b),
            .p(sf_man_prod)
        );

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;
        wire [3:0][10:0] elem_signed;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = rzr4_mag_x2(raw_a);
            assign elem_mag_b[j] = rzr4_mag_x2(raw_b);
            assign elem_sign[j] = ((raw_a == 4'h0) ? sf_a[7] : raw_a[3])
                                ^ ((raw_b == 4'h0) ? sf_b[7] : raw_b[3]);
            assign elem_valid[j] = vld_mask[i * 4 + j]
                                && (raw_a != 4'h8) && (raw_b != 4'h8)
                                && scale_valid;

            wire [10:0] elem_mag_ext = elem_valid[j] ? {3'b0, elem_mag_prod[j]} : 11'd0;
            wire [10:0] elem_neg;
            VX_ks_adder #(
                .N(11),
                .BYPASS(`FORCE_BUILTIN_ADDER(11))
            ) elem_neg_ksa (
                .dataa(~elem_mag_ext),
                .datab(11'd0),
                .cin(1'b1),
                .sum(elem_neg),
                `UNUSED_PIN(cout)
            );
            assign elem_signed[j] = elem_sign[j] ? elem_neg : elem_mag_ext;
        end

        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .USE_DSP(USE_DSP)
        ) elem_m01 (
            .clk(clk),
            .enable(enable),
            .a(elem_mag_a[1:0]),
            .b(elem_mag_b[1:0]),
            .p(elem_mag_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .USE_DSP(USE_DSP)
        ) elem_m23 (
            .clk(clk),
            .enable(enable),
            .a(elem_mag_a[3:2]),
            .b(elem_mag_b[3:2]),
            .p(elem_mag_prod[3:2])
        );

        wire [10:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N(4),
            .W(11),
            .S(11)
        ) dot_csa (
            .operands(elem_signed),
            .sum(dot_sum_vec),
            .carry(dot_carry_vec)
        );

        wire [10:0] signed_dot;
        VX_ks_adder #(
            .N(11),
            .BYPASS(`FORCE_BUILTIN_ADDER(11))
        ) dot_ksa (
            .dataa(dot_sum_vec),
            .datab(dot_carry_vec),
            .cin(1'b0),
            .sum(signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[10];
        wire [9:0] neg_dot;
        VX_ks_adder #(
            .N(10),
            .BYPASS(`FORCE_BUILTIN_ADDER(10))
        ) dot_neg_ksa (
            .dataa(~signed_dot[9:0]),
            .datab(10'd0),
            .cin(1'b1),
            .sum(neg_dot),
            `UNUSED_PIN(cout)
        );
        wire [9:0] abs_dot = dot_sign ? neg_dot : signed_dot[9:0];

        wire [17:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N(10),
            .M(8),
            .P(18),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk(clk),
            .enable(enable),
            .a(abs_dot),
            .b(sf_man_prod),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_RZR4;
        wire [EXP_TERM_W_RZR4-1:0] exp_biased =
            EXP_ADJ_RZR4 + EXP_TERM_W_RZR4'(sf_exp_a)
                          + EXP_TERM_W_RZR4'(sf_exp_b);

        VX_pipe_register #(
            .DATAW (26),
            .DEPTH (PROD_REG)
        ) pipe_result (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  ({dot_sign & ~is_zero_out, result_mag, is_zero_out}),
            .data_out ({result_sig_rzr4[i], sig_zero_rzr4[i]})
        );
        assign result_exp_rzr4[i] = is_zero_out ? '0
            : (EXP_W'(exp_biased) + EXP_W'(EXP_BASE_BIASED_RZR4));
        assign exceptions_rzr4[i].is_nan = 1'b0;
        assign exceptions_rzr4[i].is_inf = 1'b0;
        assign exceptions_rzr4[i].sign = dot_sign & ~is_zero_out;
    end
`endif  // VX_CFG_TCU_RZR4_ENABLE

    // Exponent/exception outputs join at pre-seam timing; significand and
    // sig_zero outputs join at post-seam timing.
    always_comb begin
        result_exp = '0;
        exceptions = '0;
        case (fmt_f)
        `ifdef VX_CFG_TCU_MXFP4_ENABLE
            4'(TCU_MXFP4_ID): begin
                result_exp = result_exp_mxfp4;
                exceptions = exceptions_mxfp4;
            end
        `endif
        `ifdef VX_CFG_TCU_NVFP4_ENABLE
            4'(TCU_NVFP4_ID): begin
                result_exp = result_exp_nvfp4;
                exceptions = exceptions_nvfp4;
            end
        `endif
        `ifdef VX_CFG_TCU_RZR4_ENABLE
            4'(TCU_RZR4_ID): begin
                result_exp = result_exp_rzr4;
                exceptions = exceptions_rzr4;
            end
        `endif
            default: begin
                result_exp = '0;
                exceptions = '0;
            end
        endcase
    end

    always_comb begin
        result_sig = '0;
        sig_zero   = '0;
        case (fmt_f_r)
        `ifdef VX_CFG_TCU_MXFP4_ENABLE
            4'(TCU_MXFP4_ID): begin
                result_sig = result_sig_mxfp4;
                sig_zero   = sig_zero_mxfp4;
            end
        `endif
        `ifdef VX_CFG_TCU_NVFP4_ENABLE
            4'(TCU_NVFP4_ID): begin
                result_sig = result_sig_nvfp4;
                sig_zero   = sig_zero_nvfp4;
            end
        `endif
        `ifdef VX_CFG_TCU_RZR4_ENABLE
            4'(TCU_RZR4_ID): begin
                result_sig = result_sig_rzr4;
                sig_zero   = sig_zero_rzr4;
            end
        `endif
            default: begin
                result_sig = '0;
                sig_zero   = '0;
            end
        endcase
    end

`endif  // VX_CFG_TCU_FP4_ENABLE
`endif  // VX_CFG_TCU_MX_ENABLE
endmodule
