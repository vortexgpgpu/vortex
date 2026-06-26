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
    `UNUSED_VAR ({clk, req_id, valid_in})
    `UNUSED_VAR (vld_mask)

    localparam F32_BIAS  = 127;
    localparam S_FP32    = 23;
    localparam S_SUPER   = 22;
    // f8 lanes need 2*ALIGN_SHIFT (one per sub-product)
    localparam BIAS_BASE = F32_BIAS + 2*(S_FP32 - S_SUPER) - W + WA - 1 + 128;

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam S_FP8 = VX_tcu_pkg::sign_pos(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_FP8  = 8'(BIAS_BASE - 2*B_FP8);

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam S_BF8 = VX_tcu_pkg::sign_pos(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1;
    localparam [7:0] BIAS_CONST_BF8  = 8'(BIAS_BASE - 2*B_BF8);

    for (genvar i = 0; i < TCK; ++i) begin : g_lane

        wire [1:0] lane_valid = {vld_mask[i * 4 + 2], vld_mask[i * 4 + 0]};
        wire is_bfloat = tcu_fmt_is_bfloat(fmt_f);

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

            always_comb begin
                if (is_bfloat) begin // TCU_BF8_ID (E5M2)
                    raw_ea  = 5'(raw_a[S_BF8-1 -: E_BF8]);
                    raw_eb  = 5'(raw_b[S_BF8-1 -: E_BF8]);
                    raw_ma  = {raw_a[1:0], 1'b0};
                    raw_mb  = {raw_b[1:0], 1'b0};
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                end else begin       // TCU_FP8_ID (E4M3)
                    raw_ea  = 5'(raw_a[S_FP8-1 -: E_FP8]);
                    raw_eb  = 5'(raw_b[S_FP8-1 -: E_FP8]);
                    raw_ma  = raw_a[2:0];
                    raw_mb  = raw_b[2:0];
                    raw_sa  = raw_a[7];
                    raw_sb  = raw_b[7];
                end
            end

            fedp_class_t cls_a;
            VX_tcu_tfr_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_a (
                .exp (raw_ea),
                .man (raw_ma),
                .cls (cls_a)
            );

            fedp_class_t cls_b;
            VX_tcu_tfr_classifier #(
                .EXP_W (5),
                .MAN_W (3)
            ) class_b (
                .exp (raw_eb),
                .man (raw_mb),
                .cls (cls_b)
            );

            wire is_ea_zero = (raw_ea == 0);
            wire is_eb_zero = (raw_eb == 0);

            assign ea_sel[j] = is_ea_zero ? 5'b1 : raw_ea;
            assign eb_sel[j] = is_eb_zero ? 5'b1 : raw_eb;
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

        // Pre-sort by exponent and compute both differences in parallel.
        wire [6:0] diff_0_minus_1 = {1'b0, pre_sum_0} - {1'b0, pre_sum_1};
        wire [6:0] diff_1_minus_0 = {1'b0, pre_sum_1} - {1'b0, pre_sum_0};
        `UNUSED_VAR (diff_1_minus_0[6])

        wire term0_ge_term1 = ~diff_0_minus_1[6];
        wire term0_is_max = (v0 & ~v1) || (v1 & term0_ge_term1);
        wire diff_sign = term0_is_max;

        wire [5:0] diff_abs = term0_is_max ? diff_0_minus_1[5:0] : diff_1_minus_0[5:0];
        wire [5:0] max_pre_sum = term0_is_max ? pre_sum_0 : pre_sum_1;

        wire [7:0] bias_sel = is_bfloat ? BIAS_CONST_BF8 : BIAS_CONST_FP8;

        wire [EXP_W-1:0] max_pre_sum_cpa, bias_sel_cpa;
        wire [EXP_W-1:0] final_exp;

    `ifdef VX_CFG_TCU_MX_ENABLE
        wire [3*EXP_W-1:0] sf_comp = fmt_f[3] ? {EXP_W'(sf_a), EXP_W'(sf_b), -EXP_W'(254)} : (3*EXP_W)'(0);
        VX_csa_tree #(
            .N(5),
            .W(EXP_W),
            .S(EXP_W)
        ) exp_sf_csa (
            .operands ({EXP_W'(max_pre_sum), EXP_W'(bias_sel), sf_comp}),
            .sum      (max_pre_sum_cpa),
            .carry    (bias_sel_cpa)
        );
    `else
        assign max_pre_sum_cpa = EXP_W'(max_pre_sum);
        assign bias_sel_cpa = EXP_W'(bias_sel);
    `endif
        VX_ks_adder #(
            .N(EXP_W),
            .BYPASS(`FORCE_BUILTIN_ADDER(EXP_W))
        ) exp_final_add (
            .dataa(EXP_W'(max_pre_sum_cpa)),
            .datab(EXP_W'(bias_sel_cpa)),
            .cin(1'b0),
            .sum(final_exp),
            `UNUSED_PIN(cout)
        );

        // The two fp8 mantissa products in a lane are independent unsigned 4x4
        // multiplies — small enough to pack into a single DSP48 (USE_DSP),
        // halving the f8 DSP count vs one DSP per product.
        wire [1:0][7:0] man_prod;
        VX_tcu_tfr_wmul2 #(
            .N(4),
            .USE_DSP(USE_DSP)
        ) wtmul2 (
            .a0(ma_sel[0]), .b0(mb_sel[0]),
            .a1(ma_sel[1]), .b1(mb_sel[1]),
            .p0(man_prod[0]),
            .p1(man_prod[1])
        );

        wire [7:0] man_prod0_v = man_prod[0] & {8{lane_valid[0]}};
        wire [7:0] man_prod1_v = man_prod[1] & {8{lane_valid[1]}};

        wire [7:0] prod_max = diff_sign ? man_prod0_v : man_prod1_v;
        wire [7:0] prod_min = diff_sign ? man_prod1_v : man_prod0_v;
        wire       sign_max = diff_sign ? sign_sel[0] : sign_sel[1];
        wire       sign_min = diff_sign ? sign_sel[1] : sign_sel[0];

        // Alignment
        wire [23:0] sig_max = {prod_max, 16'b0};
        wire diff_ge_32 = diff_abs[5];
        wire [4:0] shamt = diff_abs[4:0];
        wire [23:0] sig_min_shifted = diff_ge_32 ? 24'b0 : ({prod_min, 16'b0} >> shamt);

        // Add/sub reduction
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

        // Scaling by 1 avoids renormalization.
        wire [23:0] sig_add = sig_add_raw[24:1];
        `UNUSED_VAR (sig_add_raw[0])

        wire pre_sum_eq = (pre_sum_0 == pre_sum_1);
        wire mag_is_equal = pre_sum_eq && (man_prod0_v == man_prod1_v);
        wire is_zero_out = do_sub && mag_is_equal;

        wire sig_sign_raw = sub_neg ? sign_min : sign_max;
        wire sig_sign = is_zero_out ? 1'b0 : sig_sign_raw;

        assign result_sig[i] = {sig_sign, sig_add};
        assign result_exp[i] = ((v0 || v1) && !is_zero_out) ? final_exp : '0;

        // Exception merging
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
