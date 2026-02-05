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

module VX_tcu_drl_shared_mul import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [3:0]           fmt_s,

    input wire [N-1:0][31:0]   a_row,
    input wire [N-1:0][31:0]   b_col,
    input wire [31:0]          c_val,

    input fedp_class_t [N-1:0]     cls_tf32 [2],
    input fedp_class_t [TCK-1:0]   cls_fp16 [2],
    input fedp_class_t [TCK-1:0]   cls_bf16 [2],
    input fedp_class_t [2*TCK-1:0] cls_fp8 [2],
    input fedp_class_t [2*TCK-1:0] cls_bf8 [2],
    input fedp_class_t             cls_c,

    input wire [TCK-1:0][5:0]  exp_diff_f8,

    output logic [TCK:0][24:0] y
);
    `UNUSED_VAR (vld_mask)

    logic fmt_is_signed_int = tcu_fmt_is_signed_int(fmt_s[2:0]);
    logic tmt_is_bfloat = tcu_fmt_is_bfloat(fmt_s[2:0]);

    wire [TCK-1:0][10:0] man_a_tf32, man_b_tf32, man_a_fp16, man_b_fp16, man_a_bf16, man_b_bf16;
    wire [TCK-1:0]       sign_tf32, sign_fp16, sign_bf16;

    // 1. TF32 Preparation (1 op every even TCK slice)
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin : g_even_lane
            fedp_class_t ca = cls_tf32[0][i/2];
            fedp_class_t cb = cls_tf32[1][i/2];
            `UNUSED_VAR ({ca, cb})
            assign sign_tf32[i]  = a_row[i/2][18] ^ b_col[i/2][18];
            assign man_a_tf32[i] = ca.is_zero ? 11'd0 : { !ca.is_sub, a_row[i/2][9:0] };
            assign man_b_tf32[i] = cb.is_zero ? 11'd0 : { !cb.is_sub, b_col[i/2][9:0] };
        end else begin : g_odd_lane
            assign sign_tf32[i] = 1'b0;
            assign man_a_tf32[i] = 11'd0;
            assign man_b_tf32[i] = 11'd0;
        end
    end

    // 2. FP16 Preparation (1 op per TCK slice)
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp16
        fedp_class_t ca = cls_fp16[0][i];
        fedp_class_t cb = cls_fp16[1][i];
        wire [15:0] va = i[0] ? a_row[i/2][31:16] : a_row[i/2][15:0];
        wire [15:0] vb = i[0] ? b_col[i/2][31:16] : b_col[i/2][15:0];
        `UNUSED_VAR ({ca, cb, va, vb})
        assign sign_fp16[i]  = va[15] ^ vb[15];
        assign man_a_fp16[i] = ca.is_zero ? 11'd0 : { !ca.is_sub, va[9:0] };
        assign man_b_fp16[i] = cb.is_zero ? 11'd0 : { !cb.is_sub, vb[9:0] };
    end

    // 3. BF16 Preparation (1 op per TCK slice)
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf16
        fedp_class_t ca = cls_bf16[0][i];
        fedp_class_t cb = cls_bf16[1][i];
        wire [15:0] va = i[0] ? a_row[i/2][31:16] : a_row[i/2][15:0];
        wire [15:0] vb = i[0] ? b_col[i/2][31:16] : b_col[i/2][15:0];
        `UNUSED_VAR ({ca, cb, va, vb})
        assign sign_bf16[i]  = va[15] ^ vb[15];
        assign man_a_bf16[i] = ca.is_zero ? 11'd0 : { 3'd0, !ca.is_sub, va[6:0] };
        assign man_b_bf16[i] = cb.is_zero ? 11'd0 : { 3'd0, !cb.is_sub, vb[6:0] };
    end

    // 4. FP8 / BF8 Preparation (2 ops per TCK slice)
    wire [TCK-1:0][1:0][3:0] man_a_fp8, man_b_fp8, man_a_bf8, man_b_bf8;
    wire [TCK-1:0][1:0]      sign_fp8, sign_bf8;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_f8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            // Global index for class array
            localparam idx = i * 2 + j;
            // Byte access within 32-bit words
            wire [7:0] va = a_row[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] vb = b_col[i/2][(i%2)*16 + j*8 +: 8];

            // FP8 (E4M3): Sign[7], Exp[6:3] (4 bits), Man[2:0] (3 bits)
            // Extracted mantissa {1.mmm} -> 4 bits
            fedp_class_t ca_fp8 = cls_fp8[0][idx];
            fedp_class_t cb_fp8 = cls_fp8[1][idx];
            assign sign_fp8[i][j]  = va[7] ^ vb[7];
            assign man_a_fp8[i][j] = ca_fp8.is_zero ? 4'd0 : {!ca_fp8.is_sub, va[2:0]};
            assign man_b_fp8[i][j] = cb_fp8.is_zero ? 4'd0 : {!cb_fp8.is_sub, vb[2:0]};

            // BF8 (E5M2): Sign[7], Exp[6:2] (5 bits), Man[1:0] (2 bits)
            // Extracted mantissa {1.mm} -> 3 bits, padded to 4
            fedp_class_t ca_bf8 = cls_bf8[0][idx];
            fedp_class_t cb_bf8 = cls_bf8[1][idx];
            assign sign_bf8[i][j]  = va[7] ^ vb[7];
            assign man_a_bf8[i][j] = ca_bf8.is_zero ? 4'd0 : {1'b0, !ca_bf8.is_sub, va[1:0]};
            assign man_b_bf8[i][j] = cb_bf8.is_zero ? 4'd0 : {1'b0, !cb_bf8.is_sub, vb[1:0]};

            `UNUSED_VAR({ca_fp8, cb_fp8, ca_bf8, cb_bf8, va, vb})
        end
    end

    // 5. Int8 Preparation (2 ops per TCK slice)
    wire [TCK-1:0][1:0][7:0] man_a_i8, man_b_i8;
    wire [TCK-1:0][1:0]      sign_i8;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_i8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            wire [7:0] va = a_row[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] vb = b_col[i/2][(i%2)*16 + j*8 +: 8];
            wire [7:0] abs_a = fmt_is_signed_int ? (va[7] ? -va : va) : va;
            wire [7:0] abs_b = fmt_is_signed_int ? (vb[7] ? -vb : vb) : vb;
            assign man_a_i8[i][j] = abs_a;
            assign man_b_i8[i][j] = abs_b;
            assign sign_i8[i][j]  = fmt_is_signed_int && (va[7] ^ vb[7]);
        end
    end

    // 6. Int4 Preparation (4 ops per TCK slice)
    wire [TCK-1:0][3:0][3:0] man_a_i4, man_b_i4;
    wire [TCK-1:0][3:0]      sign_i4;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_i4
        for (genvar j = 0; j < 4; ++j) begin : g_sub
            wire [3:0] va = a_row[i/2][(i%2)*16 + j*4 +: 4];
            wire [3:0] vb = b_col[i/2][(i%2)*16 + j*4 +: 4];
            wire [3:0] abs_a = fmt_is_signed_int ? (va[3] ? -va : va) : va;
            wire [3:0] abs_b = fmt_is_signed_int ? (vb[3] ? -vb : vb) : vb;
            assign man_a_i4[i][j] = abs_a;
            assign man_b_i4[i][j] = abs_b;
            assign sign_i4[i][j]  = fmt_is_signed_int && (va[3] ^ vb[3]);
        end
    end

    // 7. Multiply, Align & Merge
    for (genvar i = 0; i < TCK; ++i) begin : g_mul
        // Shared TF32/FP16/BF16 Multiplier
        logic [10:0] man_a_f16, man_b_f16;
        logic        sign_f16;
        wire [21:0]  y_raw_f16;

        always_comb begin
            case(fmt_s[3:0])
                TCU_TF32_ID: begin man_a_f16 = man_a_tf32[i]; man_b_f16 = man_b_tf32[i]; sign_f16 = sign_tf32[i]; end
                TCU_FP16_ID: begin man_a_f16 = man_a_fp16[i]; man_b_f16 = man_b_fp16[i]; sign_f16 = sign_fp16[i]; end
                TCU_BF16_ID: begin man_a_f16 = man_a_bf16[i]; man_b_f16 = man_b_bf16[i]; sign_f16 = sign_bf16[i]; end
                default:     begin man_a_f16 = '0;            man_b_f16 = '0;            sign_f16 = 0;            end
            endcase
        end

        VX_wallace_mul #(
            .N(11)
        ) wtmul_f16 (
            .a(man_a_f16),
            .b(man_b_f16),
            .p(y_raw_f16)
        );

        // Shared FP8/BF8 Multiplier
        wire [1:0][7:0] y_raw_f8;
        wire [1:0]      sign_f8;

        for (genvar j = 0; j < 2; ++j) begin : g_f8
            wire [3:0] ma_f8 = tmt_is_bfloat ? man_a_bf8[i][j] : man_a_fp8[i][j];
            wire [3:0] mb_f8 = tmt_is_bfloat? man_b_bf8[i][j] : man_b_fp8[i][j];
            assign sign_f8[j] = tmt_is_bfloat ? sign_bf8[i][j] : sign_fp8[i][j];

            VX_wallace_mul #(
                .N(4)
            ) wtmul_f8 (
                .a(ma_f8),
                .b(mb_f8),
                .p(y_raw_f8[j])
            );
        end

        // Alignment & Adder for FP8/BF8
        wire [7:0] y_f8_low  = tmt_is_bfloat ? {y_raw_f8[0][5:0], 2'd0} : y_raw_f8[0];
        wire [7:0] y_f8_high = tmt_is_bfloat ? {y_raw_f8[1][5:0], 2'd0} : y_raw_f8[1];
        wire [22:0] aligned_sig_low  = exp_diff_f8[i][5] ? {y_f8_low, 15'd0} : {y_f8_low, 15'd0} >> exp_diff_f8[i][4:0];
        wire [22:0] aligned_sig_high = exp_diff_f8[i][5] ? {y_f8_high, 15'd0} >> exp_diff_f8[i][4:0] : {y_f8_high, 15'd0};
        // -- Determine which operand is larger
        wire mag_0_is_larger = (mag_0 > mag_1);
        wire [23:0] mag_0 = {1'b0, aligned_sig_low};
        wire [23:0] mag_1 = {1'b0, aligned_sig_high};
        wire [23:0] op_a = mag_0_is_larger ? mag_0 : mag_1;
        wire [23:0] op_b = mag_0_is_larger ? mag_1 : mag_0;
        // -- Absolute value adder
        wire do_sub = sign_f8[0] ^ sign_f8[1];
        wire [23:0] adder_result;
        VX_ks_adder #(
            .N(24)
        ) sig_adder_f8 (
            .cin   (do_sub),
            .dataa (op_a),
            .datab (do_sub ? ~op_b : op_b),
            .sum   (adder_result),
            `UNUSED_PIN (cout)
        );
        wire [23:0] y_f8_add = adder_result[23:0];
        wire sign_f8_add = mag_0_is_larger ? sign_f8[0] : sign_f8[1];

        // Shared I8/U8 Multiplier
        wire [1:0][16:0] y_prod_i8;
        for (genvar j = 0; j < 2; ++j) begin : g_i8
            wire [15:0] prod;
            VX_wallace_mul #(
                .N(8)
            ) wtmul_i8 (
                .a(man_a_i8[i][j]),
                .b(man_b_i8[i][j]),
                .p(prod)
            );
            wire [15:0] abs_prod = sign_i8[i][j] ? -prod : prod;
            assign y_prod_i8[j] = fmt_is_signed_int ? {abs_prod[15], abs_prod} : {1'b0, prod};
        end

        wire [16:0] y_i8_add_res;
        VX_ks_adder #(
            .N(17)
        ) i8_adder (
            .cin   (0),
            .dataa (y_prod_i8[0]),
            .datab (y_prod_i8[1]),
            .sum   (y_i8_add_res),
            `UNUSED_PIN(cout)
        );

        // Shared I4/U4 Multiplier
        wire [3:0][9:0] y_prod_i4;
        for (genvar j = 0; j < 4; ++j) begin : g_i4
            wire [7:0] prod;
            VX_wallace_mul #(
                .N(4)
            ) wtmul_i4 (
                .a(man_a_i4[i][j]),
                .b(man_b_i4[i][j]),
                .p(prod)
            );
            wire [7:0] abs_prod = sign_i4[i][j] ? -prod : prod;
            assign y_prod_i4[j] = fmt_is_signed_int ? {{2{abs_prod[7]}}, abs_prod} : {2'd0, prod};
        end

        wire [9:0] y_i4_add_res;
        VX_csa_tree #(
            .N (4),
            .W (10),
            .S (10)
        ) i4_adder (
            .operands (y_prod_i4),
            .sum      (y_i4_add_res),
            `UNUSED_PIN (cout)
        );

        // Mux Output
        always_comb begin
            case (fmt_s[3:0])
                TCU_TF32_ID, TCU_FP16_ID: y[i] = {sign_f16, y_raw_f16, 2'd0};
                TCU_BF16_ID:              y[i] = {sign_f16, y_raw_f16[15:0], 8'd0};
            `ifdef TCU_FP8_ENABLE
                TCU_FP8_ID, TCU_BF8_ID:   y[i] = {sign_f8_add, y_f8_add};
            `endif
                TCU_I8_ID:                y[i] = 25'($signed(y_i8_add_res));
                TCU_U8_ID:                y[i] = {8'd0, y_i8_add_res};
                TCU_I4_ID:                y[i] = 25'($signed(y_i4_add_res));
                TCU_U4_ID:                y[i] = {15'd0, y_i4_add_res};
                default:                  y[i] = 'x;
            endcase
        end
        `UNUSED_VAR ({sign_f8_add, y_f8_add})
    end

    // 8. C-Term Processing
    `UNUSED_VAR ({c_val[30:25], cls_c})
    assign y[TCK] = fmt_s[3] ? c_val[24:0] : {c_val[31], 1'b1, c_val[22:0]};

endmodule
