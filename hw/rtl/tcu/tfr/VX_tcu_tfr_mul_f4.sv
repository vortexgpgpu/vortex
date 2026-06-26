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
    parameter USE_DSP = 0   // map mantissa multipliers onto DSP48 slices
) (
    input wire                      clk,
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

    output logic [TCK-1:0][24:0]      result_sig,
    output logic [TCK-1:0][EXP_W-1:0] result_exp,
    output fedp_excep_t [TCK-1:0]     exceptions
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_SPARAM (W)
    `UNUSED_VAR ({clk, req_id, valid_in, fmt_f})

`ifdef VX_CFG_TCU_MX_ENABLE
`ifdef VX_CFG_TCU_FP4_ENABLE

`ifdef VX_CFG_TCU_MXFP4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_mxfp4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_mxfp4;
    fedp_excep_t [TCK-1:0]    exceptions_mxfp4;

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
            assign term_mag_shifted[j] = term_valid[j] ? (24'(f4_man_prod[j]) << SIG_SHIFT_MXFP4) : 24'd0;
        end

        // Pack the four 2x2 mantissa products into two DSP48s (two per DSP).
        VX_tcu_tfr_wmul2 #(.N(2), .USE_DSP(USE_DSP)) f4m01 (
            .a0(a_man[0]), .b0(b_man[0]), .a1(a_man[1]), .b1(b_man[1]), .p0(f4_man_prod[0]), .p1(f4_man_prod[1]));
        VX_tcu_tfr_wmul2 #(.N(2), .USE_DSP(USE_DSP)) f4m23 (
            .a0(a_man[2]), .b0(b_man[2]), .a1(a_man[3]), .b1(b_man[3]), .p0(f4_man_prod[2]), .p1(f4_man_prod[3]));

        wire [EXP_TERM_W_MXFP4-1:0] max_exp_01 = (term_exp_biased[0] >= term_exp_biased[1]) ? term_exp_biased[0] : term_exp_biased[1];
        wire [EXP_TERM_W_MXFP4-1:0] max_exp_23 = (term_exp_biased[2] >= term_exp_biased[3]) ? term_exp_biased[2] : term_exp_biased[3];
        wire [EXP_TERM_W_MXFP4-1:0] max_exp_biased = (max_exp_01 >= max_exp_23) ? max_exp_01 : max_exp_23;

        wire [3:0][26:0] term_signed;
        for (genvar j = 0; j < 4; ++j) begin : g_align
            wire [EXP_TERM_W_MXFP4-1:0] shift_amt;
            VX_ks_adder #(
                .N(EXP_TERM_W_MXFP4),
                .BYPASS(`FORCE_BUILTIN_ADDER(EXP_TERM_W_MXFP4))
            ) shift_ksa (
                .dataa(max_exp_biased),
                .datab(~term_exp_biased[j]),
                .cin(1'b1),
                .sum(shift_amt),
                `UNUSED_PIN(cout)
            );

            wire [23:0] aligned_mag = (shift_amt >= EXP_TERM_W_MXFP4'(24)) ? 24'd0 : (term_mag_shifted[j] >> shift_amt[4:0]);
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

            assign term_signed[j] = term_sign[j] ? neg_term : aligned_ext;
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
        assign result_exp_mxfp4[i] = is_zero_out ? '0 : (EXP_W'(max_exp_biased) + EXP_W'(EXP_BASE_BIASED_MXFP4));

        assign exceptions_mxfp4[i].is_nan = 1'b0;
        assign exceptions_mxfp4[i].is_inf = 1'b0;
        assign exceptions_mxfp4[i].sign   = sum_sign & ~is_zero_out;
    end
`endif  // VX_CFG_TCU_MXFP4_ENABLE

`ifdef VX_CFG_TCU_NVFP4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_nvfp4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_nvfp4;
    fedp_excep_t [TCK-1:0]    exceptions_nvfp4;

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
            assign term_mag_shifted[j] = term_valid[j] ? (24'(term_man_prod[j][10:0]) << SIG_SHIFT) : 24'd0;
        end

        // Pack the four 2x2 mantissa products into two DSP48s.
        VX_tcu_tfr_wmul2 #(.N(2), .USE_DSP(USE_DSP)) f4m01 (
            .a0(a_man[0]), .b0(b_man[0]), .a1(a_man[1]), .b1(b_man[1]), .p0(f4_man_prod[0]), .p1(f4_man_prod[1]));
        VX_tcu_tfr_wmul2 #(.N(2), .USE_DSP(USE_DSP)) f4m23 (
            .a0(a_man[2]), .b0(b_man[2]), .a1(a_man[3]), .b1(b_man[3]), .p0(f4_man_prod[2]), .p1(f4_man_prod[3]));

        // Each term scales its 2x2 product by the SHARED per-lane scale-factor
        // mantissa (sf_man_prod) -> shared-operand packing, two terms per DSP48.
        VX_tcu_tfr_wmul2s #(.NA(4), .NB(8), .USE_DSP(USE_DSP)) tm01 (
            .a0(f4_man_prod[0]), .a1(f4_man_prod[1]), .b(sf_man_prod), .p0(term_man_prod[0]), .p1(term_man_prod[1]));
        VX_tcu_tfr_wmul2s #(.NA(4), .NB(8), .USE_DSP(USE_DSP)) tm23 (
            .a0(f4_man_prod[2]), .a1(f4_man_prod[3]), .b(sf_man_prod), .p0(term_man_prod[2]), .p1(term_man_prod[3]));

        wire [5:0] max_exp_01 = (term_exp_biased[0] >= term_exp_biased[1]) ? term_exp_biased[0] : term_exp_biased[1];
        wire [5:0] max_exp_23 = (term_exp_biased[2] >= term_exp_biased[3]) ? term_exp_biased[2] : term_exp_biased[3];
        wire [5:0] max_exp_biased = (max_exp_01 >= max_exp_23) ? max_exp_01 : max_exp_23;

        wire [3:0][26:0] term_signed;
        for (genvar j = 0; j < 4; ++j) begin : g_align
            wire [5:0] shift_amt;
            VX_ks_adder #(
                .N(6),
                .BYPASS(`FORCE_BUILTIN_ADDER(6))
            ) shift_ksa (
                .dataa(max_exp_biased),
                .datab(~term_exp_biased[j]),
                .cin(1'b1),
                .sum(shift_amt),
                `UNUSED_PIN(cout)
            );

            wire [23:0] aligned_mag = (shift_amt >= 6'd24) ? 24'd0 : (term_mag_shifted[j] >> shift_amt[4:0]);
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

            assign term_signed[j] = term_sign[j] ? neg_term : aligned_ext;
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
        assign result_exp_nvfp4[i] = is_zero_out ? '0 : (EXP_W'(max_exp_biased) + EXP_W'(EXP_BASE_BIASED));

        assign exceptions_nvfp4[i].is_nan = 1'b0;
        assign exceptions_nvfp4[i].is_inf = 1'b0;
        assign exceptions_nvfp4[i].sign   = sum_sign & ~is_zero_out;
    end
`endif  // VX_CFG_TCU_NVFP4_ENABLE

    always_comb begin
        result_sig = '0;
        result_exp = '0;
        exceptions = '0;
        case (fmt_f)
        `ifdef VX_CFG_TCU_MXFP4_ENABLE
            4'(TCU_MXFP4_ID): begin
                result_sig = result_sig_mxfp4;
                result_exp = result_exp_mxfp4;
                exceptions = exceptions_mxfp4;
            end
        `endif
        `ifdef VX_CFG_TCU_NVFP4_ENABLE
            4'(TCU_NVFP4_ID): begin
                result_sig = result_sig_nvfp4;
                result_exp = result_exp_nvfp4;
                exceptions = exceptions_nvfp4;
            end
        `endif
            default: begin
                result_sig = '0;
                result_exp = '0;
                exceptions = '0;
            end
        endcase
    end

`endif  // VX_CFG_TCU_FP4_ENABLE
`endif  // VX_CFG_TCU_MX_ENABLE
endmodule
