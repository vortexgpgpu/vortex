`include "VX_define.vh"

module VX_tcu_drl_exp_bias import VX_tcu_pkg::*;  #(
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter EXP_W = 10
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]                fmtf,

    // Raw Inputs
    input wire [N-1:0][31:0]        a_row,
    input wire [N-1:0][31:0]        b_col,
    input wire [31:0]               c_val,

    // Classification Inputs
    input fedp_class_t [N-1:0]      cls_tf32 [2],
    input fedp_class_t [TCK-1:0]    cls_fp16 [2],
    input fedp_class_t [TCK-1:0]    cls_bf16 [2],
    input fedp_class_t [2*TCK-1:0]  cls_fp8 [2],
    input fedp_class_t [2*TCK-1:0]  cls_bf8 [2],
    input fedp_class_t              cls_c,

    // Output increased to [TCK:0] to include C-term
    output wire [TCK:0][EXP_W-1:0] raw_exp_y,
    output wire [TCK-1:0][5:0]     exp_diff_f8
);
    `UNUSED_VAR({vld_mask})

    localparam F32_BIAS     = 127;
    localparam F32_SIG_BITS = 23;
    localparam MUL_WIDTH    = 22;
    localparam ALIGN_SHIFT  = F32_SIG_BITS - MUL_WIDTH; // +1

    localparam E_TF32 = VX_tcu_pkg::exp_bits(TCU_TF32_ID);
    localparam S_TF32 = VX_tcu_pkg::sign_pos(TCU_TF32_ID);

    localparam E_FP16 = VX_tcu_pkg::exp_bits(TCU_FP16_ID);
    localparam S_FP16 = VX_tcu_pkg::sign_pos(TCU_FP16_ID);
    localparam B_FP16 = (1 << (E_FP16 - 1)) - 1;

    localparam E_BF16 = VX_tcu_pkg::exp_bits(TCU_BF16_ID);
    localparam S_BF16 = VX_tcu_pkg::sign_pos(TCU_BF16_ID);
    localparam B_BF16 = (1 << (E_BF16 - 1)) - 1;

    localparam E_FP8 = VX_tcu_pkg::exp_bits(TCU_FP8_ID);
    localparam S_FP8 = VX_tcu_pkg::sign_pos(TCU_FP8_ID);
    localparam B_FP8 = (1 << (E_FP8 - 1)) - 1;

    localparam E_BF8 = VX_tcu_pkg::exp_bits(TCU_BF8_ID);
    localparam S_BF8 = VX_tcu_pkg::sign_pos(TCU_BF8_ID);
    localparam B_BF8 = (1 << (E_BF8 - 1)) - 1;

    localparam [EXP_W-1:0] BIAS_CONST_TF32 = EXP_W'(F32_BIAS + ALIGN_SHIFT - W);
    localparam [EXP_W-1:0] BIAS_CONST_FP16 = EXP_W'(F32_BIAS - 2*B_FP16 + ALIGN_SHIFT - W);
    localparam [EXP_W-1:0] BIAS_CONST_BF16 = EXP_W'(F32_BIAS - 2*B_BF16 + ALIGN_SHIFT - W);
    localparam [EXP_W-1:0] BIAS_CONST_FP8  = EXP_W'(F32_BIAS - 2*B_FP8 + ALIGN_SHIFT - W);
    localparam [EXP_W-1:0] BIAS_CONST_BF8  = EXP_W'(F32_BIAS - 2*B_BF8 + ALIGN_SHIFT - W);

    // ----------------------------------------------------------------------
    // 1. Inputs Setup
    // ----------------------------------------------------------------------

    wire [TCK-1:0][7:0] ea_tf32, eb_tf32, ea_fp16, eb_fp16, ea_bf16, eb_bf16;
    wire [TCK-1:0]      z_tf32,  z_fp16,  z_bf16;

    wire [TCK-1:0][1:0][7:0] ea_fp8, eb_fp8, ea_bf8, eb_bf8;
    wire [TCK-1:0][1:0]      z_fp8,  z_bf8;

    // --- TF32 Preparation ---
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin
            wire [31:0] ra = a_row[i/2];
            wire [31:0] rb = b_col[i/2];
            `UNUSED_VAR({ra, rb})
            assign ea_tf32[i] = cls_tf32[0][i/2].is_sub ? 8'd1 : ra[S_TF32-1 -: E_TF32];
            assign eb_tf32[i] = cls_tf32[1][i/2].is_sub ? 8'd1 : rb[S_TF32-1 -: E_TF32];
            assign z_tf32[i]  = cls_tf32[0][i/2].is_zero | cls_tf32[1][i/2].is_zero | ~vld_mask[i*4];
        end else begin
            assign ea_tf32[i] = 8'd0;
            assign eb_tf32[i] = 8'd0;
            assign z_tf32[i]  = 1'b1;
        end
    end

    // --- FP16 Preparation ---
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp16
        localparam OFF = (i % 2) * 16;
        wire [31:0] ra = a_row[i/2];
        wire [31:0] rb = b_col[i/2];
        `UNUSED_VAR({ra, rb})
        assign ea_fp16[i] = cls_fp16[0][i].is_sub ? 8'd1 : {3'd0, ra[S_FP16-1+OFF -: E_FP16]};
        assign eb_fp16[i] = cls_fp16[1][i].is_sub ? 8'd1 : {3'd0, rb[S_FP16-1+OFF -: E_FP16]};
        assign z_fp16[i]  = cls_fp16[0][i].is_zero | cls_fp16[1][i].is_zero | ~vld_mask[i*4];
    end

    // --- BF16 Preparation ---
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf16
        localparam OFF = (i % 2) * 16;
        wire [31:0] ra = a_row[i/2];
        wire [31:0] rb = b_col[i/2];
        `UNUSED_VAR({ra, rb})
        assign ea_bf16[i] = cls_bf16[0][i].is_sub ? 8'd1 : ra[S_BF16-1+OFF -: E_BF16];
        assign eb_bf16[i] = cls_bf16[1][i].is_sub ? 8'd1 : rb[S_BF16-1+OFF -: E_BF16];
        assign z_bf16[i]  = cls_bf16[0][i].is_zero | cls_bf16[1][i].is_zero | ~vld_mask[i*4];
    end

    // --- FP8 Preparation ---
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            localparam OFF = (i % 2) * 16 + j * 8;
            wire [31:0] ra = a_row[i/2];
            wire [31:0] rb = b_col[i/2];
            assign ea_fp8[i][j] = cls_fp8[0][idx].is_sub ? 8'd1 : {4'd0, ra[S_FP8-1+OFF -: E_FP8]};
            assign eb_fp8[i][j] = cls_fp8[1][idx].is_sub ? 8'd1 : {4'd0, rb[S_FP8-1+OFF -: E_FP8]};
            assign z_fp8[i][j]  = cls_fp8[0][idx].is_zero | cls_fp8[1][idx].is_zero | ~vld_mask[idx*2];
            `UNUSED_VAR({ra, rb})
        end
    end

    // --- BF8 Preparation ---
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf8
        for (genvar j = 0; j < 2; ++j) begin : g_sub
            localparam idx = i * 2 + j;
            localparam OFF = (i % 2) * 16 + j * 8;
            wire [31:0] ra = a_row[i/2];
            wire [31:0] rb = b_col[i/2];
            assign ea_bf8[i][j] = cls_bf8[0][idx].is_sub ? 8'd1 : {3'd0, ra[S_BF8-1+OFF -: E_BF8]};
            assign eb_bf8[i][j] = cls_bf8[1][idx].is_sub ? 8'd1 : {3'd0, ra[S_BF8-1+OFF -: E_BF8]};
            assign z_bf8[i][j]  = cls_bf8[0][idx].is_zero | cls_bf8[1][idx].is_zero | ~vld_mask[idx*2];
            `UNUSED_VAR({ra, rb})
        end
    end

    // ----------------------------------------------------------------------
    // 2. Product Computation
    // ----------------------------------------------------------------------

    for (genvar i = 0; i < TCK; ++i) begin : g_calc
        // Shared signals (TF32/FP16/BF16)
        logic [7:0]       ea_sel_f16, eb_sel_f16;
        logic [EXP_W-1:0] bias_sel_f16;
        logic             is_zero_f16;
        wire [EXP_W-1:0]  sum_shared;

        // Packed signals (FP8/BF8)
        logic [1:0][7:0]  ea_f8_sel, eb_f8_sel;
        logic [EXP_W-1:0] bias_f8_sel;
        logic [1:0]       is_zero_f8;
        wire [EXP_W-1:0]  sum_f8_0, sum_f8_1;

        // Mux Selection
        always_comb begin
            ea_sel_f16 = 8'd0; eb_sel_f16 = 8'd0; bias_sel_f16 = '0; is_zero_f16 = 1'b1;
            ea_f8_sel = '0; eb_f8_sel = '0; bias_f8_sel = '0; is_zero_f8 = 2'b11;

            case(fmtf)
                TCU_TF32_ID: begin
                    ea_sel_f16   = ea_tf32[i];
                    eb_sel_f16   = eb_tf32[i];
                    is_zero_f16  = z_tf32[i];
                    bias_sel_f16 = BIAS_CONST_TF32;
                end
                TCU_FP16_ID: begin
                    ea_sel_f16   = ea_fp16[i];
                    eb_sel_f16   = eb_fp16[i];
                    is_zero_f16  = z_fp16[i];
                    bias_sel_f16 = BIAS_CONST_FP16;
                end
                TCU_BF16_ID: begin
                    ea_sel_f16   = ea_bf16[i];
                    eb_sel_f16   = eb_bf16[i];
                    is_zero_f16  = z_bf16[i];
                    bias_sel_f16 = BIAS_CONST_BF16;
                end
                TCU_FP8_ID: begin
                    ea_f8_sel   = ea_fp8[i];
                    eb_f8_sel   = eb_fp8[i];
                    is_zero_f8  = z_fp8[i];
                    bias_f8_sel = BIAS_CONST_FP8;
                end
                TCU_BF8_ID: begin
                    ea_f8_sel   = ea_bf8[i];
                    eb_f8_sel   = eb_bf8[i];
                    is_zero_f8  = z_bf8[i];
                    bias_f8_sel = BIAS_CONST_BF8;
                end
                default: ;
            endcase
        end

        // 2a. Shared CSA Tree
        VX_csa_tree #(
            .N (3),
            .W (EXP_W),
            .S (EXP_W)
        ) exp_adder_f16 (
            .operands ({bias_sel_f16, EXP_W'(eb_sel_f16), EXP_W'(ea_sel_f16)}),
            .sum      (sum_shared),
            `UNUSED_PIN (cout)
        );

        // 2b. Packed CSA Trees for 8-bit formats
        VX_csa_tree #(
            .N (3),
            .W (EXP_W),
            .S (EXP_W)
        ) exp_adder_f8_0 (
            .operands ({bias_f8_sel, EXP_W'(eb_f8_sel[0]), EXP_W'(ea_f8_sel[0])}),
            .sum      (sum_f8_0),
            `UNUSED_PIN (cout)
        );

        VX_csa_tree #(
            .N (3),
            .W (EXP_W),
            .S (EXP_W)
        ) exp_adder_f8_1 (
            .operands ({bias_f8_sel, EXP_W'(eb_f8_sel[1]), EXP_W'(ea_f8_sel[1])}),
            .sum      (sum_f8_1),
            `UNUSED_PIN (cout)
        );

        // 2c. Difference calculation for alignment
        wire [EXP_W-1:0] diff_f8 = sum_f8_1 - sum_f8_0;
        wire diff_f8_sign = diff_f8[EXP_W-1];
        assign exp_diff_f8[i] = {diff_f8_sign, diff_f8[4:0]};

        // Final Output Mux
        always_comb begin
            case(fmtf)
                TCU_TF32_ID, TCU_FP16_ID, TCU_BF16_ID: begin
                    raw_exp_y[i]      = is_zero_f16 ? EXP_W'(0) : sum_shared;
                end
                TCU_FP8_ID, TCU_BF8_ID: begin
                    raw_exp_y[i] = diff_f8_sign ?
                        (is_zero_f8[0] ? EXP_W'(0) : sum_f8_0) :
                        (is_zero_f8[1] ? EXP_W'(0) : sum_f8_1);
                end
                default: begin
                    raw_exp_y[i] = 'x;
                end
            endcase
        end
    end

    // ----------------------------------------------------------------------
    // 3. C-Term Exponent
    // ----------------------------------------------------------------------

    // Corrected to include Window Adjustment: c_exp - (W - 1)
    `UNUSED_VAR ({c_val[31], c_val[23:0], cls_c})
    wire [7:0] c_exp_raw = c_val[30:23];
    wire [EXP_W-1:0] c_exp_adj = EXP_W'(c_exp_raw) - EXP_W'(W-1);
    assign raw_exp_y[TCK] = cls_c.is_zero ? EXP_W'(0) : c_exp_adj;

endmodule
