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

    function automatic [3:0] e2m1_mag_x2(input logic [3:0] raw);
        case (raw)
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

`ifdef VX_CFG_TCU_MXFP4_ENABLE
    wire [TCK-1:0][24:0]      result_sig_mxfp4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_mxfp4;
    fedp_excep_t [TCK-1:0]    exceptions_mxfp4;
    wire [TCK-1:0]            sig_zero_mxfp4;

    localparam F32_BIAS_MXFP4  = 127;
    localparam S_FP32_MXFP4    = 23;
    localparam S_SUPER_MXFP4   = 22;
    localparam BIAS_BASE_MXFP4 = F32_BIAS_MXFP4 + 2*(S_FP32_MXFP4 - S_SUPER_MXFP4) - W + WA - 1 + 128;

    localparam EXP_TERM_W_MXFP4    = 6;
    localparam EXP_TERM_BIAS_MXFP4 = 1 << (EXP_TERM_W_MXFP4 - 1);
    localparam EXP_COMP_MXFP4      = -(2 + 10);
    localparam [EXP_TERM_W_MXFP4-1:0] EXP_ADJ_MXFP4 =
        EXP_TERM_W_MXFP4'(EXP_TERM_BIAS_MXFP4 + EXP_COMP_MXFP4 + 4);
    localparam [EXP_W-1:0] EXP_BASE_BIASED_MXFP4 = EXP_W'(BIAS_BASE_MXFP4 + EXP_COMP_MXFP4);
    localparam SIG_SHIFT_MXFP4 = 7;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_mxfp4
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;
        wire [3:0] elem_sign_r, elem_valid_r;
        VX_pipe_register #(
            .DATAW (8),
            .DEPTH (PROD_REG)
        ) pipe_ctrl (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  ({elem_sign, elem_valid}),
            .data_out ({elem_sign_r, elem_valid_r})
        );
        wire [3:0][10:0] elem_signed;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;

            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = e2m1_mag_x2(raw_a);
            assign elem_mag_b[j] = e2m1_mag_x2(raw_b);
            assign elem_sign[j] = raw_a[3] ^ raw_b[3];
            assign elem_valid[j] = lane_valid && (raw_a[2:0] != 3'd0)
                                         && (raw_b[2:0] != 3'd0);

            wire [10:0] elem_mag_ext = elem_valid_r[j] ? {3'b0, elem_mag_prod[j]} : 11'd0;
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
            assign elem_signed[j] = elem_sign_r[j] ? elem_neg : elem_mag_ext;
        end

        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .OUT_REG(PROD_REG),
            .USE_DSP(USE_DSP)
        ) elem_m01 (
            .clk    (clk),
            .enable (enable),
            .a(elem_mag_a[1:0]),
            .b(elem_mag_b[1:0]),
            .p(elem_mag_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .OUT_REG(PROD_REG),
            .USE_DSP(USE_DSP)
        ) elem_m23 (
            .clk    (clk),
            .enable (enable),
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
        wire is_zero_out = ~|abs_dot;
        wire signed [9:0] sf_exp_a = $signed({1'b0, sf_a}) - 10'sd127;
        wire signed [9:0] sf_exp_b = $signed({1'b0, sf_b}) - 10'sd127;
        wire signed [EXP_TERM_W_MXFP4-1:0] exp_biased = EXP_TERM_W_MXFP4'(
            10'(EXP_ADJ_MXFP4) + sf_exp_a + sf_exp_b);
        wire [23:0] result_mag = 24'(abs_dot) << SIG_SHIFT_MXFP4;

        assign result_sig_mxfp4[i] = {dot_sign & ~is_zero_out, result_mag};
        assign sig_zero_mxfp4[i] = is_zero_out;
        assign result_exp_mxfp4[i] = EXP_W'(exp_biased) + EXP_W'(EXP_BASE_BIASED_MXFP4);

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

    localparam SF_EXP_BIAS     = 7;
    localparam SF_MAN_BITS     = 3;
    localparam EXP_TERM_W      = 6;
    localparam EXP_TERM_BIAS   = 1 << (EXP_TERM_W - 1);
    localparam EXP_COMP_NVFP4  = -(2 + 2 * (SF_EXP_BIAS + SF_MAN_BITS));
    localparam [EXP_TERM_W-1:0] EXP_ADJ_NVFP4 =
        EXP_TERM_W'(EXP_TERM_BIAS + EXP_COMP_NVFP4 + 4);
    localparam [EXP_W-1:0] EXP_BASE_BIASED = EXP_W'(BIAS_BASE + EXP_COMP_NVFP4);
    localparam SIG_SHIFT_NVFP4 = 7;

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

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;
        wire [3:0][10:0] elem_signed;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;

            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = e2m1_mag_x2(raw_a);
            assign elem_mag_b[j] = e2m1_mag_x2(raw_b);
            assign elem_sign[j] = raw_a[3] ^ raw_b[3];
            assign elem_valid[j] = lane_valid && (raw_a[2:0] != 3'd0)
                                         && (raw_b[2:0] != 3'd0);

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
            .clk    (clk),
            .enable (enable),
            .a(elem_mag_a[1:0]),
            .b(elem_mag_b[1:0]),
            .p(elem_mag_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .USE_DSP(USE_DSP)
        ) elem_m23 (
            .clk    (clk),
            .enable (enable),
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
            .OUT_REG(PROD_REG),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk    (clk),
            .enable (enable),
            .a(abs_dot),
            .b(sf_man_prod),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_NVFP4;
        wire [EXP_TERM_W-1:0] exp_biased =
            EXP_ADJ_NVFP4 + EXP_TERM_W'(sf_exp_a) + EXP_TERM_W'(sf_exp_b);

        wire dot_sign_r;
        VX_pipe_register #(
            .DATAW (1),
            .DEPTH (PROD_REG)
        ) pipe_sign (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  (dot_sign),
            .data_out (dot_sign_r)
        );
        assign result_sig_nvfp4[i] = {dot_sign_r & ~is_zero_out, result_mag};
        assign sig_zero_nvfp4[i] = is_zero_out;
        assign result_exp_nvfp4[i] = EXP_W'(exp_biased) + EXP_W'(EXP_BASE_BIASED);

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

`ifdef VX_CFG_TCU_IF4_ENABLE
    wire [TCK-1:0][24:0] result_sig_if4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_if4;
    fedp_excep_t [TCK-1:0] exceptions_if4;
    wire [TCK-1:0] sig_zero_if4;

    wire [3:0] if4_exp_a = (sf_a[6:3] == 0) ? 4'd1 : sf_a[6:3];
    wire [3:0] if4_exp_b = (sf_b[6:3] == 0) ? 4'd1 : sf_b[6:3];
    wire [3:0] if4_man_a = {|sf_a[6:3], sf_a[2:0]};
    wire [3:0] if4_man_b = {|sf_b[6:3], sf_b[2:0]};
    wire [7:0] if4_man_prod;
    VX_tcu_tfr_wmul #(
        .N       (4),
        .USE_DSP (USE_DSP)
    ) if4_sf_mul (
        .clk    (clk),
        .enable (enable),
        .a      (if4_man_a),
        .b      (if4_man_b),
        .p      (if4_man_prod)
    );
    wire [2:0] if4_scale_lz;
    VX_lzc #(
        .N (8)
    ) if4_scale_lzc (
        .data_in   (if4_man_prod),
        .data_out  (if4_scale_lz),
        `UNUSED_PIN (valid_out)
    );
    wire [7:0] if4_man_norm = if4_man_prod << if4_scale_lz;

    // FP elements decode at twice their value; integers retain their magnitude.
    // Fold 1, 12/7 or 144/49 into the shared scale, with 25 fractional bits.
    wire [26:0] if4_range = (sf_a[7] && sf_b[7]) ? 27'd98608943
                         : (sf_a[7] || sf_b[7]) ? 27'd57521883 : 27'd33554432;
    wire [34:0] if4_scale_prod;
    VX_tcu_tfr_wmul #(
        .N       (27),
        .M       (8),
        .USE_DSP (USE_DSP)
    ) if4_range_mul (
        .clk    (clk),
        .enable (enable),
        .a      (if4_range),
        .b      (if4_man_norm),
        .p      (if4_scale_prod)
    );
    // Normalize the shared scale so small magnitudes retain fractional precision.
    wire if4_scale_round = if4_scale_prod[7] && ((|if4_scale_prod[6:0]) || if4_scale_prod[8]);
    wire [26:0] if4_scale_w = if4_scale_prod[34:8] + 27'(if4_scale_round);
    wire [1:0] if4_range_shift = if4_scale_w[26] ? 2'd2 : if4_scale_w[25] ? 2'd1 : 2'd0;
    wire [26:0] if4_scale_shifted = if4_scale_w >> if4_range_shift;
    wire [24:0] if4_scale = if4_scale_shifted[24:0];
    `UNUSED_VAR (if4_scale_shifted[26:25])
    localparam IF4_EXP_BASE = 127 + 2*(23 - 22) - W + WA - 1 + 128 - 44 + 32 + 4 + 1;
    wire [EXP_W-1:0] if4_exp = EXP_W'(IF4_EXP_BASE) + EXP_W'(if4_exp_a)
                           + EXP_W'(if4_exp_b) - EXP_W'(if4_scale_lz) + EXP_W'(if4_range_shift);

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_if4
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;
        wire [3:0][10:0] elem_signed;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = sf_a[7] ? (raw_a[3] ? (4'd0 - raw_a) : raw_a) : e2m1_mag_x2(raw_a);
            assign elem_mag_b[j] = sf_b[7] ? (raw_b[3] ? (4'd0 - raw_b) : raw_b) : e2m1_mag_x2(raw_b);
            assign elem_sign[j] = raw_a[3] ^ raw_b[3];
            assign elem_valid[j] = vld_mask[i * 4 + j];

            wire [10:0] elem_mag_ext = elem_valid[j] ? {3'b0, elem_mag_prod[j]} : 11'd0;
            wire [10:0] elem_neg;
            VX_ks_adder #(
                .N       (11),
                .BYPASS  (`FORCE_BUILTIN_ADDER(11))
            ) elem_neg_ksa (
                .dataa   (~elem_mag_ext),
                .datab   (11'd0),
                .cin     (1'b1),
                .sum     (elem_neg),
                `UNUSED_PIN(cout)
            );
            assign elem_signed[j] = elem_sign[j] ? elem_neg : elem_mag_ext;
        end

        VX_tcu_tfr_wmul #(
            .N       (4),
            .LANES   (2),
            .USE_DSP (USE_DSP)
        ) elem_m01 (
            .clk     (clk),
            .enable  (enable),
            .a       (elem_mag_a[1:0]),
            .b       (elem_mag_b[1:0]),
            .p       (elem_mag_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N       (4),
            .LANES   (2),
            .USE_DSP (USE_DSP)
        ) elem_m23 (
            .clk     (clk),
            .enable  (enable),
            .a       (elem_mag_a[3:2]),
            .b       (elem_mag_b[3:2]),
            .p       (elem_mag_prod[3:2])
        );

        wire [10:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N       (4),
            .W       (11),
            .S       (11)
        ) dot_csa (
            .operands(elem_signed),
            .sum     (dot_sum_vec),
            .carry   (dot_carry_vec)
        );

        wire [10:0] signed_dot;
        VX_ks_adder #(
            .N       (11),
            .BYPASS  (`FORCE_BUILTIN_ADDER(11))
        ) dot_ksa (
            .dataa   (dot_sum_vec),
            .datab   (dot_carry_vec),
            .cin     (1'b0),
            .sum     (signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[10];
        wire [9:0] neg_dot;
        VX_ks_adder #(
            .N       (10),
            .BYPASS  (`FORCE_BUILTIN_ADDER(10))
        ) dot_neg_ksa (
            .dataa   (~signed_dot[9:0]),
            .datab   (10'd0),
            .cin     (1'b1),
            .sum     (neg_dot),
            `UNUSED_PIN(cout)
        );
        wire [9:0] abs_dot = dot_sign ? neg_dot : signed_dot[9:0];

        wire [3:0] dot_lz;
        VX_lzc #(
            .N (10)
        ) dot_lzc (
            .data_in   (abs_dot),
            .data_out  (dot_lz),
            `UNUSED_PIN (valid_out)
        );
        wire [9:0] dot_norm = abs_dot << dot_lz;
        wire [34:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N       (25),
            .M       (10),
            .USE_DSP (USE_DSP),
            .OUT_REG (PROD_REG)
        ) scale_mul (
            .clk    (clk),
            .enable (enable),
            .a      (if4_scale),
            .b      (dot_norm),
            .p      (scaled_mag)
        );
        wire dot_sign_r;
        VX_pipe_register #(
            .DATAW (1),
            .DEPTH (PROD_REG)
        ) pipe_sign (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  (dot_sign),
            .data_out (dot_sign_r)
        );
        wire round_up = scaled_mag[10] && ((|scaled_mag[9:0]) || scaled_mag[11]);
        wire [23:0] result_mag = scaled_mag[34:11] + 24'(round_up);
        wire is_zero_out = ~|result_mag;
        assign result_sig_if4[i] = {dot_sign_r & ~is_zero_out, result_mag};
        assign sig_zero_if4[i] = is_zero_out;
        assign result_exp_if4[i] = if4_exp - EXP_W'(dot_lz);
        assign exceptions_if4[i].is_nan = (sf_a[6:0] == 7'h7f) || (sf_b[6:0] == 7'h7f);
        assign exceptions_if4[i].is_inf = 1'b0;
        assign exceptions_if4[i].sign = 1'b0;
    end
`endif  // VX_CFG_TCU_IF4_ENABLE

`ifdef VX_CFG_TCU_LNSF4_ENABLE
    // NVFP4 with the e4m3 block scale replaced by an LNS8 (Q4.3 two's
    // complement) scale: element format, dot-product tree and special-value
    // handling are bit-identical to NVFP4 (see VX_CFG_TCU_NVFP4_ENABLE above).
    // Only the scale-factor combine changes: log2(scale_a)+log2(scale_b) is a
    // plain fixed-point add instead of an e4m3 mantissa multiply, so the two
    // VX_tcu_tfr_wmul sf_wtmul instances NVFP4 needs to fold sf_man_a*sf_man_b
    // collapse into one adder plus an 8-entry antilog lookup for the summed
    // fractional octave (the integer octave feeds the exponent directly).
    wire [TCK-1:0][24:0]      result_sig_lnsf4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_lnsf4;
    fedp_excep_t [TCK-1:0]    exceptions_lnsf4;
    wire [TCK-1:0]            sig_zero_lnsf4;

    localparam F32_BIAS_LNSF4  = 127;
    localparam S_FP32_LNSF4    = 23;
    localparam S_SUPER_LNSF4   = 22;
    localparam BIAS_BASE_LNSF4 = F32_BIAS_LNSF4 + 2*(S_FP32_LNSF4 - S_SUPER_LNSF4) - W + WA - 1 + 128;
    // e_int below is already a TRUE (unbiased) combined exponent -- unlike
    // NVFP4, which sums two e4m3-biased raw codes and folds the 2*SF_EXP_BIAS
    // un-bias into its residual. NVFP4's own true-exponent-space residual is
    // +6 (BIAS_BASE + 6 + ea_true + eb_true; matches its implicit 2^13
    // pre/post-scale from sf_man_prod (2^6) and SIG_SHIFT (2^7), which the
    // antilog LUT below and SIG_SHIFT_LNSF4 mirror bit-for-bit), so that is
    // the constant to reuse here directly against e_int.
    localparam [EXP_W-1:0] EXP_BASE_BIASED_LNSF4 = EXP_W'(BIAS_BASE_LNSF4 + 6);
    localparam SIG_SHIFT_LNSF4 = 7;

    // round(2**(k/8) * 64), k = 0..7 -- linear mantissa for the summed
    // fractional octave, in the same UQ2.6-style convention as NVFP4's
    // sf_man_prod so scale_mul needs no width changes.
    function automatic [7:0] lnsf4_antilog(input logic [2:0] frac);
        case (frac)
            3'h0: return 8'd64;
            3'h1: return 8'd70;
            3'h2: return 8'd76;
            3'h3: return 8'd83;
            3'h4: return 8'd91;
            3'h5: return 8'd99;
            3'h6: return 8'd108;
            default: return 8'd117;
        endcase
    endfunction

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_lnsf4
        localparam K_WORD = i / 2;
        `UNUSED_VAR ({sf_a[7], sf_b[7]})

        // sf_a/sf_b[6:0]: signed two's complement log2(scale), Q4.3 (LSB = 1/8).
        // Manual sign-extension (replicate bit 6) instead of $signed()/N'(...):
        // avoids sv2v having to synthesize a cast-helper function for the
        // nested width-cast-of-a-signed-cast pattern (a construct that, in
        // hw/rtl/tcu/tfr/VX_tcu_tfr_align.sv, was seen tripping a pre-existing
        // OpenSTA Verilog-reader incompatibility with a Yosys-flattened
        // escaped identifier -- unrelated to lnsf4 and not fully root-caused;
        // see the yosys+OpenSTA synthesis note in the commit history).
        wire signed [7:0] e_a = {sf_a[6], sf_a[6:0]};
        wire signed [7:0] e_b = {sf_b[6], sf_b[6:0]};
        wire signed [7:0] e_sum = e_a + e_b;
        wire signed [4:0] e_int = e_sum >>> 3;   // floor(e_sum / 8): integer octaves
        wire [2:0] e_frac = e_sum[2:0];          // fractional remainder, always >= 0
        wire [7:0] sf_scale = lnsf4_antilog(e_frac);

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;
        wire [3:0][10:0] elem_signed;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;

            wire lane_valid = vld_mask[i * 4 + j];
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = e2m1_mag_x2(raw_a);
            assign elem_mag_b[j] = e2m1_mag_x2(raw_b);
            assign elem_sign[j] = raw_a[3] ^ raw_b[3];
            assign elem_valid[j] = lane_valid && (raw_a[2:0] != 3'd0)
                                         && (raw_b[2:0] != 3'd0);

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
            .clk    (clk),
            .enable (enable),
            .a(elem_mag_a[1:0]),
            .b(elem_mag_b[1:0]),
            .p(elem_mag_prod[1:0])
        );
        VX_tcu_tfr_wmul #(
            .N(4),
            .LANES(2),
            .USE_DSP(USE_DSP)
        ) elem_m23 (
            .clk    (clk),
            .enable (enable),
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
            .OUT_REG(PROD_REG),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk    (clk),
            .enable (enable),
            .a(abs_dot),
            .b(sf_scale),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_LNSF4;

        wire dot_sign_r;
        VX_pipe_register #(
            .DATAW (1),
            .DEPTH (PROD_REG)
        ) pipe_sign (
            .clk      (clk),
            .reset    (1'b0),
            .enable   (enable),
            .data_in  (dot_sign),
            .data_out (dot_sign_r)
        );
        assign result_sig_lnsf4[i] = {dot_sign_r & ~is_zero_out, result_mag};
        assign sig_zero_lnsf4[i] = is_zero_out;
        assign result_exp_lnsf4[i] = EXP_W'(e_int) + EXP_BASE_BIASED_LNSF4;

        // The sign field is only consumed for infinity lanes downstream.
        assign exceptions_lnsf4[i].is_nan = 1'b0;
        assign exceptions_lnsf4[i].is_inf = 1'b0;
        assign exceptions_lnsf4[i].sign   = 1'b0;
    end
`endif  // VX_CFG_TCU_LNSF4_ENABLE

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
        `ifdef VX_CFG_TCU_IF4_ENABLE
            4'(TCU_IF4_ID): begin
                result_exp = result_exp_if4;
                exceptions = exceptions_if4;
            end
        `endif
        `ifdef VX_CFG_TCU_LNSF4_ENABLE
            4'(TCU_LNSF4_ID): begin
                result_exp = result_exp_lnsf4;
                exceptions = exceptions_lnsf4;
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
        `ifdef VX_CFG_TCU_IF4_ENABLE
            4'(TCU_IF4_ID): begin
                result_sig = result_sig_if4;
                sig_zero   = sig_zero_if4;
            end
        `endif
        `ifdef VX_CFG_TCU_LNSF4_ENABLE
            4'(TCU_LNSF4_ID): begin
                result_sig = result_sig_lnsf4;
                sig_zero   = sig_zero_lnsf4;
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
