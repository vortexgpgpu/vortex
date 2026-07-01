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

module VX_tcu_tet_mul_f8 import VX_tcu_pkg::*;
#(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10
) (
    input wire                      clk,
    input wire                      reset,
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

    output logic [TCK-1:0][24:0]      result_sig,
    output logic [TCK-1:0][EXP_W-1:0] result_exp,
    output fedp_excep_t [TCK-1:0]     exceptions
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})
    `UNUSED_VAR (vld_mask)

    localparam F32_BIAS  = 127;
    localparam S_FP32    = 23;
    localparam S_SUPER   = 22;
    localparam BIAS_BASE = F32_BIAS + 2*(S_FP32 - S_SUPER) - W + WA - 1 + 128;

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam S_FP8 = VX_tcu_pkg::sign_pos(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_FP8 = 8'(BIAS_BASE - 2*B_FP8);

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam S_BF8 = VX_tcu_pkg::sign_pos(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_BF8 = 8'(BIAS_BASE - 2*B_BF8);

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire [1:0] lane_valid = {vld_mask[i * 4 + 2], vld_mask[i * 4 + 0]};
        wire is_bfloat = tcu_fmt_is_bfloat(fmt_f);

        wire [1:0][5:0] pre_sum;
        wire [1:0][7:0] man_prod;
        wire [1:0]      zero_sel, sign_sel, nan_sel, inf_sel;

        for (genvar j = 0; j < 2; ++j) begin : g_extract
            localparam OFF = (i % 2) * 16 + j * 8;

            wire [7:0] raw_a = a_row[i/2][OFF +: 8];
            wire [7:0] raw_b = b_col[i/2][OFF +: 8];

            logic [4:0] raw_ea, raw_eb;
            logic [2:0] raw_ma, raw_mb;
            logic       raw_sa, raw_sb;

            always_comb begin
                if (is_bfloat) begin
                    raw_ea  = 5'(raw_a[S_BF8-1 -: E_BF8]);
                    raw_eb  = 5'(raw_b[S_BF8-1 -: E_BF8]);
                    raw_ma  = {raw_a[1:0], 1'b0};
                    raw_mb  = {raw_b[1:0], 1'b0};
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                end else begin
                    raw_ea  = 5'(raw_a[S_FP8-1 -: E_FP8]);
                    raw_eb  = 5'(raw_b[S_FP8-1 -: E_FP8]);
                    raw_ma  = raw_a[2:0];
                    raw_mb  = raw_b[2:0];
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                end
            end

            fedp_class_t cls_a;
            VX_tcu_tet_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_a (
                .exp (raw_ea),
                .man (raw_ma),
                .cls (cls_a)
            );

            fedp_class_t cls_b;
            VX_tcu_tet_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_b (
                .exp (raw_eb),
                .man (raw_mb),
                .cls (cls_b)
            );

            wire is_ea_zero = (raw_ea == 0);
            wire is_eb_zero = (raw_eb == 0);

            wire [4:0] ea_sel = is_ea_zero ? 5'b1 : raw_ea;
            wire [4:0] eb_sel = is_eb_zero ? 5'b1 : raw_eb;
            wire [3:0] ma_sel = {~is_ea_zero, raw_ma};
            wire [3:0] mb_sel = {~is_eb_zero, raw_mb};
            `UNUSED_VAR (cls_a.is_sub)
            `UNUSED_VAR (cls_b.is_sub)

            assign sign_sel[j] = raw_sa ^ raw_sb;
            assign zero_sel[j] = cls_a.is_zero | cls_b.is_zero;

            wire a_is_inf = is_bfloat ? cls_a.is_inf : 1'b0;
            wire b_is_inf = is_bfloat ? cls_b.is_inf : 1'b0;
            wire a_is_nan = is_bfloat ? cls_a.is_nan : (raw_ea == 5'h0F) && (raw_ma == 3'b111);
            wire b_is_nan = is_bfloat ? cls_b.is_nan : (raw_eb == 5'h0F) && (raw_mb == 3'b111);

            wire nan_in = a_is_nan | b_is_nan;
            wire inf_z  = (a_is_inf & cls_b.is_zero) | (cls_a.is_zero & b_is_inf);
            wire inf_op = a_is_inf | b_is_inf;

            assign nan_sel[j] = nan_in | inf_z;
            assign inf_sel[j] = inf_op & ~inf_z;

            VX_ks_adder #(
                .N      (6),
                .BYPASS (`FORCE_BUILTIN_ADDER(6))
            ) exp_add (
                .dataa (6'(ea_sel)),
                .datab (6'(eb_sel)),
                .cin   (1'b0),
                .sum   (pre_sum[j]),
                `UNUSED_PIN(cout)
            );

            wire [0:0][3:0] wmul_a = ma_sel;
            wire [0:0][3:0] wmul_b = mb_sel;
            wire [0:0][7:0] wmul_p;

            VX_tcu_tet_wmul #(
                .N (4),
                .P (8),
                .USE_DSP (1)
            ) wtmul (
                .a (wmul_a),
                .b (wmul_b),
                .p (wmul_p)
            );

            assign man_prod[j] = wmul_p[0];
        end

        wire v0_w = ~zero_sel[0] && lane_valid[0];
        wire v1_w = ~zero_sel[1] && lane_valid[1];

        wire [6:0] diff_0_minus_1_w = {1'b0, pre_sum[0]} - {1'b0, pre_sum[1]};
        wire [6:0] diff_1_minus_0_w = {1'b0, pre_sum[1]} - {1'b0, pre_sum[0]};
        `UNUSED_VAR (diff_1_minus_0_w[6])

        wire term0_ge_term1_w = ~diff_0_minus_1_w[6];
        wire term0_is_max_w = (v0_w & ~v1_w) || (v1_w & term0_ge_term1_w);
        wire [5:0] diff_abs_w = term0_is_max_w ? diff_0_minus_1_w[5:0] : diff_1_minus_0_w[5:0];
        wire [5:0] max_pre_sum_w = term0_is_max_w ? pre_sum[0] : pre_sum[1];
        wire pre_sum_eq_w = (pre_sum[0] == pre_sum[1]);

        wire [5:0] s1_max_pre_sum;
        wire [5:0] s1_diff_abs;
        wire [1:0][7:0] s1_man_prod;
        wire [1:0]      s1_lane_valid;
        wire [1:0]      s1_zero_sel, s1_sign_sel, s1_nan_sel, s1_inf_sel;
        wire            s1_diff_sign;
        wire            s1_pre_sum_eq;
        wire            s1_is_bfloat;
    `ifdef VX_CFG_TCU_MX_ENABLE
        wire            s1_fmt_is_mx;
        wire [7:0]      s1_sf_a, s1_sf_b;
    `endif

        VX_tcu_tet_register #(
        `ifdef VX_CFG_TCU_MX_ENABLE
            .DATAW ((2 * 6) + (2 * 8) + (4 * 2) + 2 + 2 + 1 + 1 + 16),
        `else
            .DATAW ((2 * 6) + (2 * 8) + (4 * 2) + 2 + 2 + 1),
        `endif
            .DEPTH (1)
        ) pipe_mul_s0 (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
        `ifdef VX_CFG_TCU_MX_ENABLE
            .data_in  ({max_pre_sum_w,    diff_abs_w,    man_prod,    lane_valid,    zero_sel,    sign_sel,    nan_sel,    inf_sel,    term0_is_max_w, pre_sum_eq_w,  is_bfloat,    fmt_f[3],      sf_a,    sf_b}),
            .data_out ({s1_max_pre_sum,   s1_diff_abs,   s1_man_prod, s1_lane_valid, s1_zero_sel, s1_sign_sel, s1_nan_sel, s1_inf_sel, s1_diff_sign,   s1_pre_sum_eq, s1_is_bfloat, s1_fmt_is_mx, s1_sf_a, s1_sf_b})
        `else
            .data_in  ({max_pre_sum_w,    diff_abs_w,    man_prod,    lane_valid,    zero_sel,    sign_sel,    nan_sel,    inf_sel,    term0_is_max_w, pre_sum_eq_w,  is_bfloat}),
            .data_out ({s1_max_pre_sum,   s1_diff_abs,   s1_man_prod, s1_lane_valid, s1_zero_sel, s1_sign_sel, s1_nan_sel, s1_inf_sel, s1_diff_sign,   s1_pre_sum_eq, s1_is_bfloat})
        `endif
        );

        wire v0 = ~s1_zero_sel[0] && s1_lane_valid[0];
        wire v1 = ~s1_zero_sel[1] && s1_lane_valid[1];

        wire [7:0] bias_sel = s1_is_bfloat ? BIAS_CONST_BF8 : BIAS_CONST_FP8;

        wire [EXP_W-1:0] bias_sel_cpa;
    `ifdef VX_CFG_TCU_MX_ENABLE
        wire [EXP_W-1:0] bias_cpa_sum, bias_cpa_carry;
        wire [3*EXP_W-1:0] sf_comp = s1_fmt_is_mx ? {EXP_W'(s1_sf_a), EXP_W'(s1_sf_b), -EXP_W'(254)} : (3*EXP_W)'(0);
        VX_csa_tree #(
            .N (4),
            .W (EXP_W),
            .S (EXP_W)
        ) exp_sf_csa (
            .operands ({EXP_W'(bias_sel), sf_comp}),
            .sum      (bias_cpa_sum),
            .carry    (bias_cpa_carry)
        );
        VX_ks_adder #(
            .N      (EXP_W),
            .BYPASS (`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_bias_add (
            .dataa (bias_cpa_sum),
            .datab (bias_cpa_carry),
            .cin   (1'b0),
            .sum   (bias_sel_cpa),
            `UNUSED_PIN(cout)
        );
    `else
        assign bias_sel_cpa = EXP_W'(bias_sel);
    `endif

        wire [EXP_W-1:0] final_exp;
        VX_ks_adder #(
            .N      (EXP_W),
            .BYPASS (`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_final_add (
            .dataa (EXP_W'(s1_max_pre_sum)),
            .datab (bias_sel_cpa),
            .cin   (1'b0),
            .sum   (final_exp),
            `UNUSED_PIN(cout)
        );

        wire [7:0] man_prod0_v = s1_man_prod[0] & {8{s1_lane_valid[0]}};
        wire [7:0] man_prod1_v = s1_man_prod[1] & {8{s1_lane_valid[1]}};

        wire [7:0] prod_max = s1_diff_sign ? man_prod0_v : man_prod1_v;
        wire [7:0] prod_min = s1_diff_sign ? man_prod1_v : man_prod0_v;
        wire       sign_max = s1_diff_sign ? s1_sign_sel[0] : s1_sign_sel[1];
        wire       sign_min = s1_diff_sign ? s1_sign_sel[1] : s1_sign_sel[0];

        wire [23:0] sig_max = {prod_max, 16'b0};
        wire diff_ge_32 = s1_diff_abs[5];
        wire [4:0] shamt = s1_diff_abs[4:0];
        wire [23:0] sig_min_shifted = diff_ge_32 ? 24'b0 : ({prod_min, 16'b0} >> shamt);

        wire [24:0] sig_max_ext = {1'b0, sig_max};
        wire [24:0] sig_min_ext = {1'b0, sig_min_shifted};

        wire do_sub = s1_sign_sel[0] ^ s1_sign_sel[1];

        wire [24:0] add_raw;
        VX_ks_adder #(
            .N      (25),
            .BYPASS (`FORCE_BUILTIN_ADDER(25))
        ) add_adder (
            .dataa (sig_max_ext),
            .datab (sig_min_ext),
            .cin   (1'b0),
            .sum   (add_raw),
            `UNUSED_PIN(cout)
        );

        wire [24:0] sub_raw;
        VX_ks_adder #(
            .N      (25),
            .BYPASS (`FORCE_BUILTIN_ADDER(25))
        ) sub_adder (
            .dataa (sig_max_ext),
            .datab (~sig_min_ext),
            .cin   (1'b1),
            .sum   (sub_raw),
            `UNUSED_PIN(cout)
        );

        wire [24:0] sub_rev_raw;
        VX_ks_adder #(
            .N      (25),
            .BYPASS (`FORCE_BUILTIN_ADDER(25))
        ) sub_rev_adder (
            .dataa (sig_min_ext),
            .datab (~sig_max_ext),
            .cin   (1'b1),
            .sum   (sub_rev_raw),
            `UNUSED_PIN(cout)
        );

        wire sub_neg = sub_raw[24];
        wire [24:0] sub_abs = sub_neg ? sub_rev_raw : sub_raw;
        wire [24:0] sig_add_raw = do_sub ? sub_abs : add_raw;

        wire [23:0] sig_add = sig_add_raw[24:1];
        `UNUSED_VAR (sig_add_raw[0])

        wire mag_is_equal = s1_pre_sum_eq && (man_prod0_v == man_prod1_v);
        wire is_zero_out = do_sub && mag_is_equal;

        wire sig_sign_raw = sub_neg ? sign_min : sign_max;
        wire sig_sign = is_zero_out ? 1'b0 : sig_sign_raw;

        assign result_sig[i] = {sig_sign, sig_add};
        assign result_exp[i] = ((v0 || v1) && !is_zero_out) ? final_exp : '0;

        wire pos_inf_0 = s1_inf_sel[0] && ~s1_sign_sel[0] && s1_lane_valid[0];
        wire neg_inf_0 = s1_inf_sel[0] &&  s1_sign_sel[0] && s1_lane_valid[0];
        wire pos_inf_1 = s1_inf_sel[1] && ~s1_sign_sel[1] && s1_lane_valid[1];
        wire neg_inf_1 = s1_inf_sel[1] &&  s1_sign_sel[1] && s1_lane_valid[1];

        wire add_nan = (pos_inf_0 && neg_inf_1) || (neg_inf_0 && pos_inf_1);

        wire any_nan = (|(s1_nan_sel & s1_lane_valid)) || add_nan;
        wire any_inf = (|(s1_inf_sel & s1_lane_valid)) && ~any_nan;

        wire final_sign_inf = s1_inf_sel[0] ? s1_sign_sel[0] : s1_sign_sel[1];

        assign exceptions[i].is_nan = any_nan;
        assign exceptions[i].is_inf = any_inf;
        assign exceptions[i].sign   = any_inf ? final_sign_inf : sig_sign;

    `ifdef DBG_TRACE_TCU
        always_ff @(posedge clk) begin
            if (valid_in && s1_lane_valid != 0) begin
                `TRACE(4, ("%t: %s FEDP-REDUCE(%0d) lane=%0d: ", $time, INSTANCE_ID, req_id, i));
                `TRACE(4, ("max=(%0d, 0x%0h, %0d), ", sign_max, sig_max, s1_max_pre_sum));
                `TRACE(4, ("min=(%0d, 0x%0h) ", sign_min, sig_min_shifted));
                `TRACE(4, ("-> s=%0d, P=0x%0h, E=%0d\n", sig_sign, sig_add, final_exp));
            end
        end
    `endif

    end

endmodule
