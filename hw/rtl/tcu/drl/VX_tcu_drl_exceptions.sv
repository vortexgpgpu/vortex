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

module VX_tcu_drl_exceptions import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]            fmtf,

    input fedp_class_t [N-1:0]     cls_tf32 [2],
    input fedp_class_t [TCK-1:0]   cls_fp16 [2],
    input fedp_class_t [TCK-1:0]   cls_bf16 [2],
    input fedp_class_t [2*TCK-1:0] cls_fp8 [2],
    input fedp_class_t [2*TCK-1:0] cls_bf8 [2],
    input fedp_class_t             cls_c,

    output fedp_excep_t          exceptions
);
    `UNUSED_VAR ({vld_mask, cls_c})

    // Intermediate Signals
    wire [TCK-1:0] nan_in_tf32, inf_z_tf32, inf_op_tf32, sign_tf32;
    wire [TCK-1:0] nan_in_fp16, inf_z_fp16, inf_op_fp16, sign_fp16;
    wire [TCK-1:0] nan_in_bf16, inf_z_bf16, inf_op_bf16, sign_bf16;
    wire [TCK-1:0][1:0] nan_in_fp8, inf_z_fp8, inf_op_fp8, sign_fp8;
    wire [TCK-1:0][1:0] nan_in_bf8, inf_z_bf8, inf_op_bf8, sign_bf8;

    // ----------------------------------------------------------------------
    // 1a. TF32 Preparation (Even lanes only)
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin : g_even_lane
            assign nan_in_tf32[i] = cls_tf32[0][i/2].is_nan | cls_tf32[1][i/2].is_nan;
            assign inf_z_tf32[i]  = (cls_tf32[0][i/2].is_inf & cls_tf32[1][i/2].is_zero)
                                  | (cls_tf32[0][i/2].is_zero & cls_tf32[1][i/2].is_inf);
            assign inf_op_tf32[i] = cls_tf32[0][i/2].is_inf | cls_tf32[1][i/2].is_inf;
            assign sign_tf32[i]   = cls_tf32[0][i/2].sign ^ cls_tf32[1][i/2].sign;
        end else begin : g_odd_lane
            assign nan_in_tf32[i] = 1'b0;
            assign inf_z_tf32[i]  = 1'b0;
            assign inf_op_tf32[i] = 1'b0;
            assign sign_tf32[i]   = 1'b0;
        end
    end
    `UNUSED_VAR ({nan_in_tf32, inf_z_tf32, inf_op_tf32, sign_tf32})

    // ----------------------------------------------------------------------
    // 1b. FP16 Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp16
        assign nan_in_fp16[i] = cls_fp16[0][i].is_nan | cls_fp16[1][i].is_nan;
        assign inf_z_fp16[i]  = (cls_fp16[0][i].is_inf & cls_fp16[1][i].is_zero)
                              | (cls_fp16[0][i].is_zero & cls_fp16[1][i].is_inf);
        assign inf_op_fp16[i] = cls_fp16[0][i].is_inf | cls_fp16[1][i].is_inf;
        assign sign_fp16[i]   = cls_fp16[0][i].sign ^ cls_fp16[1][i].sign;
    end

    // ----------------------------------------------------------------------
    // 1c. BF16 Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf16
        assign nan_in_bf16[i] = cls_bf16[0][i].is_nan | cls_bf16[1][i].is_nan;
        assign inf_z_bf16[i]  = (cls_bf16[0][i].is_inf & cls_bf16[1][i].is_zero)
                              | (cls_bf16[0][i].is_zero & cls_bf16[1][i].is_inf);
        assign inf_op_bf16[i] = cls_bf16[0][i].is_inf | cls_bf16[1][i].is_inf;
        assign sign_bf16[i]   = cls_bf16[0][i].sign ^ cls_bf16[1][i].sign;
    end
    `UNUSED_VAR ({nan_in_bf16, inf_z_bf16, inf_op_bf16, sign_bf16})

    // ----------------------------------------------------------------------
    // 1d. FP8 (E4M3) Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            assign nan_in_fp8[i][j] = cls_fp8[0][idx].is_nan | cls_fp8[1][idx].is_nan;
            assign inf_z_fp8[i][j]  = (cls_fp8[0][idx].is_inf & cls_fp8[1][idx].is_zero)
                                    | (cls_fp8[0][idx].is_zero & cls_fp8[1][idx].is_inf);
            assign inf_op_fp8[i][j] = cls_fp8[0][idx].is_inf | cls_fp8[1][idx].is_inf;
            assign sign_fp8[i][j]   = cls_fp8[0][idx].sign ^ cls_fp8[1][idx].sign;
        end
    end

    // ----------------------------------------------------------------------
    // 1e. BF8 (E5M2) Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            assign nan_in_bf8[i][j] = cls_bf8[0][idx].is_nan | cls_bf8[1][idx].is_nan;
            assign inf_z_bf8[i][j]  = (cls_bf8[0][idx].is_inf & cls_bf8[1][idx].is_zero)
                                    | (cls_bf8[0][idx].is_zero & cls_bf8[1][idx].is_inf);
            assign inf_op_bf8[i][j] = cls_bf8[0][idx].is_inf | cls_bf8[1][idx].is_inf;
            assign sign_bf8[i][j]   = cls_bf8[0][idx].sign ^ cls_bf8[1][idx].sign;
        end
    end

    // ----------------------------------------------------------------------
    // 2. Merge & Mask
    // ----------------------------------------------------------------------
    wire [TCK-1:0] prod_nan, prod_inf, prod_sign;

    for (genvar i = 0; i < TCK; ++i) begin : g_merge
        logic n_in, i_z, i_op, sgn, valid_lane;

        // FP8/BF8 sub-product addition inf flags
        wire [1:0] fp8_pos_inf, fp8_neg_inf, fp8_valid_inf;
        wire [1:0] bf8_pos_inf, bf8_neg_inf, bf8_valid_inf;

        for (genvar j = 0; j < 2; ++j) begin : g_inf_flags
            assign fp8_valid_inf[j] = inf_op_fp8[i][j] & ~inf_z_fp8[i][j];
            assign fp8_pos_inf[j]   = fp8_valid_inf[j] & ~sign_fp8[i][j];
            assign fp8_neg_inf[j]   = fp8_valid_inf[j] & sign_fp8[i][j];

            assign bf8_valid_inf[j] = inf_op_bf8[i][j] & ~inf_z_bf8[i][j];
            assign bf8_pos_inf[j]   = bf8_valid_inf[j] & ~sign_bf8[i][j];
            assign bf8_neg_inf[j]   = bf8_valid_inf[j] & sign_bf8[i][j];
        end

        // NaN if either product has NaN input or inf*0
        wire n_in_fp8_comb = (|nan_in_fp8[i]) | (|inf_z_fp8[i]);
        wire n_in_bf8_comb = (|nan_in_bf8[i]) | (|inf_z_bf8[i]);

        // Check for +inf + -inf in the addition (generates NaN)
        wire fp8_add_nan = (fp8_pos_inf[0] & fp8_neg_inf[1]) | (fp8_neg_inf[0] & fp8_pos_inf[1]);
        wire bf8_add_nan = (bf8_pos_inf[0] & bf8_neg_inf[1]) | (bf8_neg_inf[0] & bf8_pos_inf[1]);

        // Combined infinity check (either product is inf and no add-nan)
        wire i_op_fp8_comb = (|fp8_valid_inf);
        wire i_op_bf8_comb = (|bf8_valid_inf);

        // Sign of combined result (dominant inf sign, prefer positive if same magnitude)
        wire sgn_fp8_comb = (|fp8_neg_inf) & ~(|fp8_pos_inf);
        wire sgn_bf8_comb = (|bf8_neg_inf) & ~(|bf8_pos_inf);
        `UNUSED_VAR ({n_in_fp8_comb, n_in_bf8_comb, fp8_add_nan, bf8_add_nan, i_op_fp8_comb, i_op_bf8_comb, sgn_fp8_comb, sgn_bf8_comb})

        always_comb begin
            case (fmtf)
                TCU_FP16_ID: begin
                    n_in = nan_in_fp16[i]; i_z = inf_z_fp16[i]; i_op = inf_op_fp16[i]; sgn = sign_fp16[i];
                    valid_lane = vld_mask[i * 4];
                end
            `ifdef TCU_BF16_ENABLE
                TCU_BF16_ID: begin
                    n_in = nan_in_bf16[i]; i_z = inf_z_bf16[i]; i_op = inf_op_bf16[i]; sgn = sign_bf16[i];
                    valid_lane = vld_mask[i * 4];
                end
            `endif
            `ifdef TCU_TF32_ENABLE
                TCU_TF32_ID: begin
                    n_in = nan_in_tf32[i]; i_z = inf_z_tf32[i]; i_op = inf_op_tf32[i]; sgn = sign_tf32[i];
                    valid_lane = ((i % 2) == 0) ? vld_mask[i * 4] : 1'b0;
                end
            `endif
            `ifdef TCU_FP8_ENABLE
                TCU_FP8_ID: begin
                    n_in = n_in_fp8_comb | fp8_add_nan; i_z = 1'b0; i_op = i_op_fp8_comb; sgn = sgn_fp8_comb;
                    valid_lane = vld_mask[i * 2];
                end
                TCU_BF8_ID: begin
                    n_in = n_in_bf8_comb | bf8_add_nan; i_z = 1'b0; i_op = i_op_bf8_comb; sgn = sgn_bf8_comb;
                    valid_lane = vld_mask[i * 2];
                end
            `endif
                default: begin
                    n_in=0; i_z=0; i_op=0; sgn=0; valid_lane=0;
                end
            endcase
        end
        assign prod_nan[i]  = (n_in | i_z) & valid_lane;
        assign prod_inf[i]  = (i_op & ~i_z) & valid_lane;
        assign prod_sign[i] = sgn;
    end

    // ----------------------------------------------------------------------
    // 3. Global Aggregation
    // ----------------------------------------------------------------------

    wire any_input_nan = (|prod_nan) | cls_c.is_nan;

    wire [TCK-1:0] p_pos_inf = prod_inf & ~prod_sign;
    wire [TCK-1:0] p_neg_inf = prod_inf & prod_sign;

    wire c_pos_inf = cls_c.is_inf & ~cls_c.sign;
    wire c_neg_inf = cls_c.is_inf & cls_c.sign;

    wire has_pos = (|p_pos_inf) | c_pos_inf;
    wire has_neg = (|p_neg_inf) | c_neg_inf;

    wire add_gen_nan = has_pos & has_neg;

    wire res_nan = any_input_nan | add_gen_nan;

    assign exceptions.sign   = has_neg & ~has_pos;
    assign exceptions.is_nan = res_nan;
    assign exceptions.is_inf = (has_pos | has_neg) & ~res_nan;

endmodule
