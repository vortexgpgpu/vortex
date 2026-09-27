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
    parameter PROD_REG = 0, // product/flag register stages (multiply-stage seam)
    parameter SF    = 1     // scale slots; lanes [s*TCK/SF, (s+1)*TCK/SF) share sf_a/sf_b
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
    `UNUSED_PARAM (SF)
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

    // Lanes of one scale slot are reduced in the integer domain and scaled
    // once; see the LNS8 scale section below for the lane/slot mapping.
    `STATIC_ASSERT ((TCK % SF) == 0, ("VX_tcu_tfr_mul_f4: TCK must be a multiple of SF"))
    localparam NV_LPG   = TCK / SF;
    localparam NV_LPG_W = $clog2(NV_LPG);
    localparam NV_DOT_W = 11 + NV_LPG_W;
    localparam SIG_SHIFT_NVFP4_GRP = SIG_SHIFT_NVFP4 - NV_LPG_W;

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

    wire [EXP_TERM_W-1:0] nv_exp_biased =
        EXP_ADJ_NVFP4 + EXP_TERM_W'(sf_exp_a) + EXP_TERM_W'(sf_exp_b);
    wire [EXP_W-1:0] nv_result_exp =
        EXP_W'(nv_exp_biased) + EXP_W'(EXP_BASE_BIASED) + EXP_W'(NV_LPG_W);

    wire [TCK-1:0][3:0][NV_DOT_W-1:0] nv_elem_signed;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_nvfp4
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;

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
            wire [10:0] elem_signed = elem_sign[j] ? elem_neg : elem_mag_ext;
            assign nv_elem_signed[i][j] = {{NV_LPG_W{elem_signed[10]}}, elem_signed};
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
    end

    for (genvar g = 0; g < SF; ++g) begin : g_grp_nvfp4
        wire [NV_LPG*4-1:0][NV_DOT_W-1:0] grp_terms;
        for (genvar l = 0; l < NV_LPG; ++l) begin : g_terms
            for (genvar j = 0; j < 4; ++j) begin : g_term
                assign grp_terms[l * 4 + j] = nv_elem_signed[g * NV_LPG + l][j];
            end
        end

        wire [NV_DOT_W-1:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N(NV_LPG * 4),
            .W(NV_DOT_W),
            .S(NV_DOT_W)
        ) dot_csa (
            .operands(grp_terms),
            .sum(dot_sum_vec),
            .carry(dot_carry_vec)
        );

        wire [NV_DOT_W-1:0] signed_dot;
        VX_ks_adder #(
            .N(NV_DOT_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(NV_DOT_W))
        ) dot_ksa (
            .dataa(dot_sum_vec),
            .datab(dot_carry_vec),
            .cin(1'b0),
            .sum(signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[NV_DOT_W-1];
        wire [NV_DOT_W-2:0] neg_dot;
        VX_ks_adder #(
            .N(NV_DOT_W-1),
            .BYPASS(`FORCE_BUILTIN_ADDER(NV_DOT_W-1))
        ) dot_neg_ksa (
            .dataa(~signed_dot[NV_DOT_W-2:0]),
            .datab((NV_DOT_W-1)'(0)),
            .cin(1'b1),
            .sum(neg_dot),
            `UNUSED_PIN(cout)
        );

        wire [NV_DOT_W-2:0] abs_dot = dot_sign ? neg_dot : signed_dot[NV_DOT_W-2:0];
        wire [NV_DOT_W+6:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N(NV_DOT_W-1),
            .M(8),
            .P(NV_DOT_W+7),
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
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_NVFP4_GRP;

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

        for (genvar l = 0; l < NV_LPG; ++l) begin : g_out
            localparam I = g * NV_LPG + l;
            if (l == 0) begin : g_lead
                assign result_sig_nvfp4[I] = {dot_sign_r & ~is_zero_out, result_mag};
                assign sig_zero_nvfp4[I]   = is_zero_out;
            end else begin : g_idle
                assign result_sig_nvfp4[I] = '0;
                assign sig_zero_nvfp4[I]   = 1'b1;
            end
            assign result_exp_nvfp4[I] = nv_result_exp;

            // The sign field is only consumed for infinity lanes downstream.
            assign exceptions_nvfp4[I].is_nan = 1'b0;
            assign exceptions_nvfp4[I].is_inf = 1'b0;
            assign exceptions_nvfp4[I].sign   = 1'b0;
        end
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

    // Lanes of one scale slot are reduced in the integer domain and scaled
    // once; see the LNS8 scale section below for the lane/slot mapping.
    `STATIC_ASSERT ((TCK % SF) == 0, ("VX_tcu_tfr_mul_f4: TCK must be a multiple of SF"))
    localparam RZR4_LPG   = TCK / SF;
    localparam RZR4_LPG_W = $clog2(RZR4_LPG);
    localparam RZR4_DOT_W = 11 + RZR4_LPG_W;
    localparam SIG_SHIFT_RZR4_GRP = SIG_SHIFT_RZR4 - RZR4_LPG_W;

    wire [3:0] rzr4_sf_exp_raw_a = sf_a[6:3];
    wire [3:0] rzr4_sf_exp_raw_b = sf_b[6:3];
    wire [3:0] rzr4_sf_exp_a = (rzr4_sf_exp_raw_a == 0) ? 4'd1 : rzr4_sf_exp_raw_a;
    wire [3:0] rzr4_sf_exp_b = (rzr4_sf_exp_raw_b == 0) ? 4'd1 : rzr4_sf_exp_raw_b;
    wire [3:0] rzr4_sf_man_a = (rzr4_sf_exp_raw_a == 0) ? {1'b0, sf_a[2:0]} : {1'b1, sf_a[2:0]};
    wire [3:0] rzr4_sf_man_b = (rzr4_sf_exp_raw_b == 0) ? {1'b0, sf_b[2:0]} : {1'b1, sf_b[2:0]};
    wire rzr4_scale_valid = (|rzr4_sf_man_a) && (|rzr4_sf_man_b);

    wire [7:0] rzr4_sf_man_prod;
    VX_tcu_tfr_wmul #(
        .N(4),
        .USE_DSP(USE_DSP)
    ) rzr4_sf_wtmul (
        .clk(clk),
        .enable(enable),
        .a(rzr4_sf_man_a),
        .b(rzr4_sf_man_b),
        .p(rzr4_sf_man_prod)
    );

    wire [EXP_TERM_W_RZR4-1:0] rzr4_exp_biased =
        EXP_ADJ_RZR4 + EXP_TERM_W_RZR4'(rzr4_sf_exp_a)
                      + EXP_TERM_W_RZR4'(rzr4_sf_exp_b);
    wire [EXP_W-1:0] rzr4_result_exp = EXP_W'(rzr4_exp_biased)
        + EXP_W'(EXP_BASE_BIASED_RZR4) + EXP_W'(RZR4_LPG_W);

    wire [TCK-1:0][3:0][RZR4_DOT_W-1:0] rzr4_elem_signed;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_rzr4
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;

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
                                && rzr4_scale_valid;

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
            wire [10:0] elem_signed = elem_sign[j] ? elem_neg : elem_mag_ext;
            assign rzr4_elem_signed[i][j] = {{RZR4_LPG_W{elem_signed[10]}}, elem_signed};
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
    end

    for (genvar g = 0; g < SF; ++g) begin : g_grp_rzr4
        wire [RZR4_LPG*4-1:0][RZR4_DOT_W-1:0] grp_terms;
        for (genvar l = 0; l < RZR4_LPG; ++l) begin : g_terms
            for (genvar j = 0; j < 4; ++j) begin : g_term
                assign grp_terms[l * 4 + j] = rzr4_elem_signed[g * RZR4_LPG + l][j];
            end
        end

        wire [RZR4_DOT_W-1:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N(RZR4_LPG * 4),
            .W(RZR4_DOT_W),
            .S(RZR4_DOT_W)
        ) dot_csa (
            .operands(grp_terms),
            .sum(dot_sum_vec),
            .carry(dot_carry_vec)
        );

        wire [RZR4_DOT_W-1:0] signed_dot;
        VX_ks_adder #(
            .N(RZR4_DOT_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(RZR4_DOT_W))
        ) dot_ksa (
            .dataa(dot_sum_vec),
            .datab(dot_carry_vec),
            .cin(1'b0),
            .sum(signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[RZR4_DOT_W-1];
        wire [RZR4_DOT_W-2:0] neg_dot;
        VX_ks_adder #(
            .N(RZR4_DOT_W-1),
            .BYPASS(`FORCE_BUILTIN_ADDER(RZR4_DOT_W-1))
        ) dot_neg_ksa (
            .dataa(~signed_dot[RZR4_DOT_W-2:0]),
            .datab((RZR4_DOT_W-1)'(0)),
            .cin(1'b1),
            .sum(neg_dot),
            `UNUSED_PIN(cout)
        );
        wire [RZR4_DOT_W-2:0] abs_dot = dot_sign ? neg_dot : signed_dot[RZR4_DOT_W-2:0];

        wire [RZR4_DOT_W+6:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N(RZR4_DOT_W-1),
            .M(8),
            .P(RZR4_DOT_W+7),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk(clk),
            .enable(enable),
            .a(abs_dot),
            .b(rzr4_sf_man_prod),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_RZR4_GRP;

        for (genvar l = 0; l < RZR4_LPG; ++l) begin : g_out
            localparam I = g * RZR4_LPG + l;
            if (l == 0) begin : g_lead
                VX_pipe_register #(
                    .DATAW (26),
                    .DEPTH (PROD_REG)
                ) pipe_result (
                    .clk      (clk),
                    .reset    (1'b0),
                    .enable   (enable),
                    .data_in  ({dot_sign & ~is_zero_out, result_mag, is_zero_out}),
                    .data_out ({result_sig_rzr4[I], sig_zero_rzr4[I]})
                );
                assign result_exp_rzr4[I] = is_zero_out ? '0 : rzr4_result_exp;
                assign exceptions_rzr4[I].sign = dot_sign & ~is_zero_out;
            end else begin : g_idle
                assign result_sig_rzr4[I] = '0;
                assign sig_zero_rzr4[I]   = 1'b1;
                assign result_exp_rzr4[I] = '0;
                assign exceptions_rzr4[I].sign = 1'b0;
            end
            assign exceptions_rzr4[I].is_nan = 1'b0;
            assign exceptions_rzr4[I].is_inf = 1'b0;
        end
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
`define VX_TCU_TFR_MUL_F4_LNS
`endif
`ifdef VX_CFG_TCU_RZR4_LNS_ENABLE
`define VX_TCU_TFR_MUL_F4_LNS
`endif

`ifdef VX_TCU_TFR_MUL_F4_LNS
    // LNS8 (Q4.3 two's complement) block scale shared by LNSF4 and RZR4_LNS:
    // log2(scale_a)+log2(scale_b) is a plain fixed-point add instead of an
    // e4m3 mantissa multiply, followed by an 8-entry antilog lookup for the
    // summed fractional octave (the integer octave feeds the exponent).
    localparam F32_BIAS_LNS  = 127;
    localparam S_FP32_LNS    = 23;
    localparam S_SUPER_LNS   = 22;
    localparam BIAS_BASE_LNS = F32_BIAS_LNS + 2*(S_FP32_LNS - S_SUPER_LNS) - W + WA - 1 + 128;
    // e_int below is already a TRUE (unbiased) combined exponent -- unlike
    // NVFP4, which sums two e4m3-biased raw codes and folds the 2*SF_EXP_BIAS
    // un-bias into its residual. NVFP4's own true-exponent-space residual is
    // +6 (BIAS_BASE + 6 + ea_true + eb_true; matches its implicit 2^13
    // pre/post-scale from sf_man_prod (2^6) and SIG_SHIFT (2^7), which the
    // antilog LUT below and SIG_SHIFT_LNS mirror bit-for-bit), so that is
    // the constant to reuse here directly against e_int.
    localparam [EXP_W-1:0] EXP_BASE_BIASED_LNS = EXP_W'(BIAS_BASE_LNS + 6);

    // Every lane of a scale slot (lanes [g*LPG, (g+1)*LPG), matching
    // VX_tcu_tfr_shared_mul's SF_SLOT map) shares one block scale, so the
    // slot's element products are reduced exactly in the integer domain and
    // scaled once. The slot's sum is emitted on its first lane; the others
    // report sig_zero so the downstream max-exponent search skips them.
    `STATIC_ASSERT ((TCK % SF) == 0, ("VX_tcu_tfr_mul_f4: TCK must be a multiple of SF"))
    localparam LNS_LPG   = TCK / SF;
    localparam LNS_LPG_W = $clog2(LNS_LPG);
    localparam LNS_DOT_W = 11 + LNS_LPG_W;
    // The antilog 2^(k/8) is stored as [1.AL] fixed point, so scale_mul's
    // operand is AL+1 bits. SIG_SHIFT_LNS restores NVFP4's 2^13 pre-scale
    // for any AL and group width, keeping the exponent residual fixed.
    localparam AL = `VX_CFG_TCU_LNSF4_ANTILOG_BITS;
    `STATIC_ASSERT ((AL >= 2) && (AL <= 8), ("VX_tcu_tfr_mul_f4: VX_CFG_TCU_LNSF4_ANTILOG_BITS must be 2..8"))
    localparam SIG_SHIFT_LNS = 13 - AL - LNS_LPG_W;
    localparam [EXP_W-1:0] EXP_GRP_BIASED_LNS = EXP_BASE_BIASED_LNS + EXP_W'(LNS_LPG_W);

    // round(2**(k/8) * 2**AL), k = 0..7; every entry is < 2**(AL+1).
    function automatic [8:0] lns8_antilog(input logic [2:0] k);
        case (AL)
        2: case (k)
            3'h0: return 9'd4;  3'h1: return 9'd4;  3'h2: return 9'd5;  3'h3: return 9'd5;
            3'h4: return 9'd6;  3'h5: return 9'd6;  3'h6: return 9'd7;  default: return 9'd7;
        endcase
        3: case (k)
            3'h0: return 9'd8;  3'h1: return 9'd9;  3'h2: return 9'd10;  3'h3: return 9'd10;
            3'h4: return 9'd11;  3'h5: return 9'd12;  3'h6: return 9'd13;  default: return 9'd15;
        endcase
        4: case (k)
            3'h0: return 9'd16;  3'h1: return 9'd17;  3'h2: return 9'd19;  3'h3: return 9'd21;
            3'h4: return 9'd23;  3'h5: return 9'd25;  3'h6: return 9'd27;  default: return 9'd29;
        endcase
        5: case (k)
            3'h0: return 9'd32;  3'h1: return 9'd35;  3'h2: return 9'd38;  3'h3: return 9'd41;
            3'h4: return 9'd45;  3'h5: return 9'd49;  3'h6: return 9'd54;  default: return 9'd59;
        endcase
        6: case (k)
            3'h0: return 9'd64;  3'h1: return 9'd70;  3'h2: return 9'd76;  3'h3: return 9'd83;
            3'h4: return 9'd91;  3'h5: return 9'd99;  3'h6: return 9'd108;  default: return 9'd117;
        endcase
        7: case (k)
            3'h0: return 9'd128;  3'h1: return 9'd140;  3'h2: return 9'd152;  3'h3: return 9'd166;
            3'h4: return 9'd181;  3'h5: return 9'd197;  3'h6: return 9'd215;  default: return 9'd235;
        endcase
        default: case (k)
            3'h0: return 9'd256;  3'h1: return 9'd279;  3'h2: return 9'd304;  3'h3: return 9'd332;
            3'h4: return 9'd362;  3'h5: return 9'd395;  3'h6: return 9'd431;  default: return 9'd470;
        endcase
        endcase
    endfunction

    // sf_a/sf_b[6:0]: signed two's complement log2(scale), Q4.3 (LSB = 1/8);
    // sf[7] is not part of the scale (unused by LNSF4, RaZeR sign in RZR4_LNS).
    // Manual sign-extension instead of $signed()/N'(...) keeps sv2v from
    // emitting a cast-helper function for the nested cast.
    wire signed [7:0] lns_e_a = {sf_a[6], sf_a[6:0]};
    wire signed [7:0] lns_e_b = {sf_b[6], sf_b[6:0]};
    wire signed [7:0] lns_e_sum = lns_e_a + lns_e_b;
    wire signed [4:0] lns_e_int = lns_e_sum >>> 3;
    wire [8:0] lns_scale_full = lns8_antilog(lns_e_sum[2:0]);
    wire [AL:0] lns_scale = lns_scale_full[AL:0];
    if (AL < 8) begin : g_lns_scale_unused
        `UNUSED_VAR (lns_scale_full[8:AL+1])
    end
    wire [EXP_W-1:0] lns_result_exp = EXP_W'(lns_e_int) + EXP_GRP_BIASED_LNS;
`endif  // VX_TCU_TFR_MUL_F4_LNS

`ifdef VX_CFG_TCU_LNSF4_ENABLE
    // NVFP4 elements with the LNS8 block scale above.
    wire [TCK-1:0][24:0]      result_sig_lnsf4;
    wire [TCK-1:0][EXP_W-1:0] result_exp_lnsf4;
    fedp_excep_t [TCK-1:0]    exceptions_lnsf4;
    wire [TCK-1:0]            sig_zero_lnsf4;

    `UNUSED_VAR ({sf_a[7], sf_b[7]})

    wire [TCK-1:0][3:0][LNS_DOT_W-1:0] lns_elem_signed;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_lnsf4
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;

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
            wire [10:0] elem_signed = elem_sign[j] ? elem_neg : elem_mag_ext;
            assign lns_elem_signed[i][j] = {{LNS_LPG_W{elem_signed[10]}}, elem_signed};
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
    end

    for (genvar g = 0; g < SF; ++g) begin : g_grp_lnsf4
        wire [LNS_LPG*4-1:0][LNS_DOT_W-1:0] grp_terms;
        for (genvar l = 0; l < LNS_LPG; ++l) begin : g_terms
            for (genvar j = 0; j < 4; ++j) begin : g_term
                assign grp_terms[l * 4 + j] = lns_elem_signed[g * LNS_LPG + l][j];
            end
        end

        wire [LNS_DOT_W-1:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N(LNS_LPG * 4),
            .W(LNS_DOT_W),
            .S(LNS_DOT_W)
        ) dot_csa (
            .operands(grp_terms),
            .sum(dot_sum_vec),
            .carry(dot_carry_vec)
        );

        wire [LNS_DOT_W-1:0] signed_dot;
        VX_ks_adder #(
            .N(LNS_DOT_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(LNS_DOT_W))
        ) dot_ksa (
            .dataa(dot_sum_vec),
            .datab(dot_carry_vec),
            .cin(1'b0),
            .sum(signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[LNS_DOT_W-1];
        wire [LNS_DOT_W-2:0] neg_dot;
        VX_ks_adder #(
            .N(LNS_DOT_W-1),
            .BYPASS(`FORCE_BUILTIN_ADDER(LNS_DOT_W-1))
        ) dot_neg_ksa (
            .dataa(~signed_dot[LNS_DOT_W-2:0]),
            .datab((LNS_DOT_W-1)'(0)),
            .cin(1'b1),
            .sum(neg_dot),
            `UNUSED_PIN(cout)
        );

        wire [LNS_DOT_W-2:0] abs_dot = dot_sign ? neg_dot : signed_dot[LNS_DOT_W-2:0];
        wire [LNS_DOT_W+AL-1:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N(LNS_DOT_W-1),
            .M(AL+1),
            .P(LNS_DOT_W+AL),
            .OUT_REG(PROD_REG),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk    (clk),
            .enable (enable),
            .a(abs_dot),
            .b(lns_scale),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_LNS;

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

        for (genvar l = 0; l < LNS_LPG; ++l) begin : g_out
            localparam I = g * LNS_LPG + l;
            if (l == 0) begin : g_lead
                assign result_sig_lnsf4[I] = {dot_sign_r & ~is_zero_out, result_mag};
                assign sig_zero_lnsf4[I]   = is_zero_out;
            end else begin : g_idle
                assign result_sig_lnsf4[I] = '0;
                assign sig_zero_lnsf4[I]   = 1'b1;
            end
            assign result_exp_lnsf4[I] = lns_result_exp;

            // The sign field is only consumed for infinity lanes downstream.
            assign exceptions_lnsf4[I].is_nan = 1'b0;
            assign exceptions_lnsf4[I].is_inf = 1'b0;
            assign exceptions_lnsf4[I].sign   = 1'b0;
        end
    end
`endif  // VX_CFG_TCU_LNSF4_ENABLE

`ifdef VX_CFG_TCU_RZR4_LNS_ENABLE
    // RZR4 (RaZeR) elements with the LNS8 block scale above; sf[7] still
    // selects the sign of the special +/-5 code, and an LNS scale is never zero.
    wire [TCK-1:0][24:0]      result_sig_rzr4_lns;
    wire [TCK-1:0]            sig_zero_rzr4_lns;
    wire [TCK-1:0][EXP_W-1:0] result_exp_rzr4_lns;
    fedp_excep_t [TCK-1:0]    exceptions_rzr4_lns;

    wire [TCK-1:0][3:0][LNS_DOT_W-1:0] rzl_elem_signed;

    for (genvar i = 0; i < TCK; ++i) begin : g_lane_rzr4_lns
        localparam K_WORD = i / 2;

        wire [3:0][3:0] elem_mag_a, elem_mag_b;
        wire [3:0][7:0] elem_mag_prod;
        wire [3:0] elem_sign;
        wire [3:0] elem_valid;

        for (genvar j = 0; j < 4; ++j) begin : g_term
            localparam OFF = (i % 2) * 16 + j * 4;
            wire [3:0] raw_a = a_row[K_WORD][OFF +: 4];
            wire [3:0] raw_b = b_col[K_WORD][OFF +: 4];

            assign elem_mag_a[j] = rzr4_mag_x2(raw_a);
            assign elem_mag_b[j] = rzr4_mag_x2(raw_b);
            assign elem_sign[j] = ((raw_a == 4'h0) ? sf_a[7] : raw_a[3])
                                ^ ((raw_b == 4'h0) ? sf_b[7] : raw_b[3]);
            assign elem_valid[j] = vld_mask[i * 4 + j]
                                && (raw_a != 4'h8) && (raw_b != 4'h8);

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
            wire [10:0] elem_signed = elem_sign[j] ? elem_neg : elem_mag_ext;
            assign rzl_elem_signed[i][j] = {{LNS_LPG_W{elem_signed[10]}}, elem_signed};
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
    end

    for (genvar g = 0; g < SF; ++g) begin : g_grp_rzr4_lns
        wire [LNS_LPG*4-1:0][LNS_DOT_W-1:0] grp_terms;
        for (genvar l = 0; l < LNS_LPG; ++l) begin : g_terms
            for (genvar j = 0; j < 4; ++j) begin : g_term
                assign grp_terms[l * 4 + j] = rzl_elem_signed[g * LNS_LPG + l][j];
            end
        end

        wire [LNS_DOT_W-1:0] dot_sum_vec, dot_carry_vec;
        VX_csa_tree #(
            .N(LNS_LPG * 4),
            .W(LNS_DOT_W),
            .S(LNS_DOT_W)
        ) dot_csa (
            .operands(grp_terms),
            .sum(dot_sum_vec),
            .carry(dot_carry_vec)
        );

        wire [LNS_DOT_W-1:0] signed_dot;
        VX_ks_adder #(
            .N(LNS_DOT_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(LNS_DOT_W))
        ) dot_ksa (
            .dataa(dot_sum_vec),
            .datab(dot_carry_vec),
            .cin(1'b0),
            .sum(signed_dot),
            `UNUSED_PIN(cout)
        );

        wire dot_sign = signed_dot[LNS_DOT_W-1];
        wire [LNS_DOT_W-2:0] neg_dot;
        VX_ks_adder #(
            .N(LNS_DOT_W-1),
            .BYPASS(`FORCE_BUILTIN_ADDER(LNS_DOT_W-1))
        ) dot_neg_ksa (
            .dataa(~signed_dot[LNS_DOT_W-2:0]),
            .datab((LNS_DOT_W-1)'(0)),
            .cin(1'b1),
            .sum(neg_dot),
            `UNUSED_PIN(cout)
        );
        wire [LNS_DOT_W-2:0] abs_dot = dot_sign ? neg_dot : signed_dot[LNS_DOT_W-2:0];

        wire [LNS_DOT_W+AL-1:0] scaled_mag;
        VX_tcu_tfr_wmul #(
            .N(LNS_DOT_W-1),
            .M(AL+1),
            .P(LNS_DOT_W+AL),
            .USE_DSP(USE_DSP)
        ) scale_mul (
            .clk(clk),
            .enable(enable),
            .a(abs_dot),
            .b(lns_scale),
            .p(scaled_mag)
        );

        wire is_zero_out = ~|scaled_mag;
        wire [23:0] result_mag = 24'(scaled_mag) << SIG_SHIFT_LNS;

        for (genvar l = 0; l < LNS_LPG; ++l) begin : g_out
            localparam I = g * LNS_LPG + l;
            if (l == 0) begin : g_lead
                VX_pipe_register #(
                    .DATAW (26),
                    .DEPTH (PROD_REG)
                ) pipe_result (
                    .clk      (clk),
                    .reset    (1'b0),
                    .enable   (enable),
                    .data_in  ({dot_sign & ~is_zero_out, result_mag, is_zero_out}),
                    .data_out ({result_sig_rzr4_lns[I], sig_zero_rzr4_lns[I]})
                );
                assign result_exp_rzr4_lns[I] = is_zero_out ? '0 : lns_result_exp;
                assign exceptions_rzr4_lns[I].sign = dot_sign & ~is_zero_out;
            end else begin : g_idle
                assign result_sig_rzr4_lns[I] = '0;
                assign sig_zero_rzr4_lns[I]   = 1'b1;
                assign result_exp_rzr4_lns[I] = '0;
                assign exceptions_rzr4_lns[I].sign = 1'b0;
            end
            assign exceptions_rzr4_lns[I].is_nan = 1'b0;
            assign exceptions_rzr4_lns[I].is_inf = 1'b0;
        end
    end
`endif  // VX_CFG_TCU_RZR4_LNS_ENABLE

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
        `ifdef VX_CFG_TCU_RZR4_LNS_ENABLE
            4'(TCU_RZR4_LNS_ID): begin
                result_exp = result_exp_rzr4_lns;
                exceptions = exceptions_rzr4_lns;
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
        `ifdef VX_CFG_TCU_RZR4_LNS_ENABLE
            4'(TCU_RZR4_LNS_ID): begin
                result_sig = result_sig_rzr4_lns;
                sig_zero   = sig_zero_rzr4_lns;
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

`undef VX_TCU_TFR_MUL_F4_LNS
