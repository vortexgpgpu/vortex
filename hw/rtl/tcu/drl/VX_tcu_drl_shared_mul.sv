
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

    input wire [TCK-1:0]       exp_low_larger,
    input wire [TCK-1:0][6:0]  raw_exp_diff,

    output logic [TCK:0][24:0] y
);
    `UNUSED_VAR ({vld_mask, exp_low_larger, raw_exp_diff})

    wire [TCK-1:0][10:0] man_a_tf32, man_b_tf32, man_a_fp16, man_b_fp16, man_a_bf16, man_b_bf16;
    wire [TCK-1:0]       sign_tf32, sign_fp16, sign_bf16;

    // 1. TF32 Preparation
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin
            fedp_class_t ca = cls_tf32[0][i/2];
            fedp_class_t cb = cls_tf32[1][i/2];
            `UNUSED_VAR ({ca, cb})
            assign sign_tf32[i]  = a_row[i/2][31] ^ b_col[i/2][31];
            assign man_a_tf32[i] = ca.is_zero ? 11'd0 : { !ca.is_sub, a_row[i/2][22:13] };
            assign man_b_tf32[i] = cb.is_zero ? 11'd0 : { !cb.is_sub, b_col[i/2][22:13] };
        end else begin
            assign sign_tf32[i] = 1'b0;
            assign man_a_tf32[i] = 11'd0;
            assign man_b_tf32[i] = 11'd0;
        end
    end

    // 2. FP16 Preparation
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

    // 3. BF16 Preparation
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

            // FP8 (E5M2): 1S, 5E, 2M. Extracted mantissa {1.mm} -> 3 bits. Padded to 4.
            fedp_class_t ca_fp8 = cls_fp8[0][idx];
            fedp_class_t cb_fp8 = cls_fp8[1][idx];
            assign sign_fp8[i][j]  = va[7] ^ vb[7];
            assign man_a_fp8[i][j] = ca_fp8.is_zero ? 4'd0 : {1'b0, !ca_fp8.is_sub, va[1:0]};
            assign man_b_fp8[i][j] = cb_fp8.is_zero ? 4'd0 : {1'b0, !cb_fp8.is_sub, vb[1:0]};

            // BF8 (E4M3): 1S, 4E, 3M. Extracted mantissa {1.mmm} -> 4 bits.
            fedp_class_t ca_bf8 = cls_bf8[0][idx];
            fedp_class_t cb_bf8 = cls_bf8[1][idx];
            assign sign_bf8[i][j]  = va[7] ^ vb[7];
            assign man_a_bf8[i][j] = ca_bf8.is_zero ? 4'd0 : {!ca_bf8.is_sub, va[2:0]};
            assign man_b_bf8[i][j] = cb_bf8.is_zero ? 4'd0 : {!cb_bf8.is_sub, vb[2:0]};

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
            wire signed_op = (fmt_s == 4'(TCU_I8_ID));
            // If signed, extract abs. If unsigned, value is abs.
            wire [7:0] abs_a = signed_op ? (va[7] ? -va : va) : va;
            wire [7:0] abs_b = signed_op ? (vb[7] ? -vb : vb) : vb;
            assign man_a_i8[i][j] = abs_a;
            assign man_b_i8[i][j] = abs_b;
            assign sign_i8[i][j]  = signed_op ? (va[7] ^ vb[7]) : 1'b0;
        end
    end

    // 6. Int4 Preparation (4 ops per TCK slice)
    wire [TCK-1:0][3:0][3:0] man_a_i4, man_b_i4;
    wire [TCK-1:0][3:0]      sign_i4;
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_i4
        for (genvar j = 0; j < 4; ++j) begin : g_sub
            wire [3:0] va = a_row[i/2][(i%2)*16 + j*4 +: 4];
            wire [3:0] vb = b_col[i/2][(i%2)*16 + j*4 +: 4];
            wire signed_op = (fmt_s == 4'(TCU_I4_ID));
            wire [3:0] abs_a = signed_op ? (va[3] ? -va : va) : va;
            wire [3:0] abs_b = signed_op ? (vb[3] ? -vb : vb) : vb;
            assign man_a_i4[i][j] = abs_a;
            assign man_b_i4[i][j] = abs_b;
            assign sign_i4[i][j]  = signed_op ? (va[3] ^ vb[3]) : 1'b0;
        end
    end

    // 7. Multiply, Align & Merge
    for (genvar i = 0; i < TCK; ++i) begin : g_mul
        // Shared TF32/FP16/BF16 Multiplier
        logic [10:0] man_a_sh, man_b_sh;
        logic        sign_sh;
        wire [21:0]  y_raw_sh;

        always_comb begin
            case(fmt_s[3:0])
                TCU_TF32_ID: begin man_a_sh = man_a_tf32[i]; man_b_sh = man_b_tf32[i]; sign_sh = sign_tf32[i]; end
                TCU_FP16_ID: begin man_a_sh = man_a_fp16[i]; man_b_sh = man_b_fp16[i]; sign_sh = sign_fp16[i]; end
                TCU_BF16_ID: begin man_a_sh = man_a_bf16[i]; man_b_sh = man_b_bf16[i]; sign_sh = sign_bf16[i]; end
                default:     begin man_a_sh = '0;            man_b_sh = '0;            sign_sh = 0;            end
            endcase
        end

        VX_wallace_mul #(
            .N(11)
        ) wtmul_tf32_f16_vf16 (
            .a(man_a_sh),
            .b(man_b_sh),
            .p(y_raw_sh)
        );

        // Shared FP8/BF8 Multiplier
        wire [1:0][7:0] y_raw_f8;
        wire [1:0]      sign_f8_curr;

        for (genvar j = 0; j < 2; ++j) begin : g_f8
            wire [3:0] ma_f8 = (fmt_s == 4'(TCU_FP8_ID)) ? man_a_fp8[i][j] : man_a_bf8[i][j];
            wire [3:0] mb_f8 = (fmt_s == 4'(TCU_FP8_ID)) ? man_b_fp8[i][j] : man_b_bf8[i][j];
            assign sign_f8_curr[j] = (fmt_s == 4'(TCU_FP8_ID)) ? sign_fp8[i][j] : sign_bf8[i][j];

            VX_wallace_mul #(
                .N(4)
            ) wtmul_f8_bf8 (
                .a(ma_f8),
                .b(mb_f8),
                .p(y_raw_f8[j])
            );
        end

        // Alignment & Adder for FP8/BF8
        wire [6:0] shift_amt_f8 = exp_low_larger[i] ? -raw_exp_diff[i] : raw_exp_diff[i];
        wire [7:0] y_f8_low  = (fmt_s == 4'(TCU_FP8_ID)) ? y_raw_f8[0] : {y_raw_f8[0][5:0], 2'd0};
        wire [7:0] y_f8_high = (fmt_s == 4'(TCU_FP8_ID)) ? y_raw_f8[1] : {y_raw_f8[1][5:0], 2'd0};

        wire [22:0] aligned_sig_low  = exp_low_larger[i] ? {y_f8_low, 15'd0} : {y_f8_low, 15'd0} >> shift_amt_f8;
        wire [22:0] aligned_sig_high = exp_low_larger[i] ? {y_f8_high, 15'd0} >> shift_amt_f8 : {y_f8_high, 15'd0};

        wire [23:0] signed_sig_low  = sign_f8_curr[0] ? -aligned_sig_low  : {1'b0, aligned_sig_low};
        wire [23:0] signed_sig_high = sign_f8_curr[1] ? -aligned_sig_high : {1'b0, aligned_sig_high};

        wire [24:0] signed_sig_res;
        VX_ks_adder #(
            .N(24)
        ) sig_adder_f8 (
            .dataa (signed_sig_low),
            .datab (signed_sig_high),
            .sum   (signed_sig_res[23:0]),
            .cout  (signed_sig_res[24])
        );

        wire sign_f8_add = signed_sig_res[24];
        wire [23:0] y_f8_add = sign_f8_add ? -signed_sig_res[23:0] : signed_sig_res[23:0];

        // Shared I8/U8 Multiplier
        wire [1:0][15:0] y_abs_i8;
        wire [1:0][16:0] y_signed_i8;

        for (genvar j = 0; j < 2; ++j) begin : g_i8
            VX_wallace_mul #(
                .N(8)
            ) wtmul_i8 (
                .a(man_a_i8[i][j]),
                .b(man_b_i8[i][j]),
                .p(y_abs_i8[j])
            );
            wire [15:0] s_prod = sign_i8[i][j] ? -y_abs_i8[j] : y_abs_i8[j];
            assign y_signed_i8[j] = (fmt_s == 4'(TCU_I8_ID)) ? {s_prod[15], s_prod} : {1'b0, y_abs_i8[j]};
        end

        wire [16:0] y_i8_add_res;
        VX_ks_adder #(
            .N(17)
        ) i8_adder (
            .dataa (y_signed_i8[0]),
            .datab (y_signed_i8[1]),
            .sum   (y_i8_add_res),
            `UNUSED_PIN(cout)
        );

        // Shared I4/U4 Multiplier
        wire [3:0][7:0] y_abs_i4;
        wire [3:0][9:0] y_signed_i4;
        wire [9:0]      y_i4_add_res;

        for (genvar j = 0; j < 4; ++j) begin : g_i4
            VX_wallace_mul #(
                .N(4)
            ) wtmul_i4 (
                .a(man_a_i4[i][j]),
                .b(man_b_i4[i][j]),
                .p(y_abs_i4[j])
            );
            wire [7:0] s_prod = sign_i4[i][j] ? -y_abs_i4[j] : y_abs_i4[j];
            assign y_signed_i4[j] = (fmt_s == 4'(TCU_I4_ID)) ? {{2{s_prod[7]}}, s_prod} : {2'd0, y_abs_i4[j]};
        end

        VX_csa_tree #(
            .N (4),
            .W (10),
            .S (10)
        ) i4_adder (
            .operands (y_signed_i4),
            .sum      (y_i4_add_res),
            `UNUSED_PIN (cout)
        );

        // Mux Output
        always_comb begin
            case (fmt_s[3:0])
                TCU_TF32_ID, TCU_FP16_ID: y[i] = {sign_sh, y_raw_sh, 2'd0};
                TCU_BF16_ID:              y[i] = {sign_sh, y_raw_sh[15:0], 8'd0};
                TCU_FP8_ID, TCU_BF8_ID:   y[i] = {sign_f8_add, y_f8_add};
                TCU_I8_ID:                y[i] = 25'($signed(y_i8_add_res));
                TCU_U8_ID:                y[i] = {8'd0, y_i8_add_res};
                TCU_I4_ID:                y[i] = 25'($signed(y_i4_add_res));
                TCU_U4_ID:                y[i] = {15'd0, y_i4_add_res};
                default:                  y[i] = 'x;
            endcase
        end
    end

    // 8. C-Term Processing
    `UNUSED_VAR ({c_val[30:25], cls_c})
    assign y[TCK] = fmt_s[3] ? c_val[24:0] : {c_val[31], 1'b1, c_val[22:0]};

endmodule
