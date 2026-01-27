`include "VX_define.vh"

module VX_tcu_drl_exceptions import VX_tcu_pkg::*; #(
    parameter N   = 2,
    parameter TCK = 2 * N
) (
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input wire [2:0]            fmtf,

    input fedp_class_t [N-1:0]   cls_tf32 [2],
    input fedp_class_t [TCK-1:0] cls_fp16 [2],
    input fedp_class_t [TCK-1:0] cls_bf16 [2],
    input fedp_class_t           cls_c,

    output fedp_excep_t          exceptions
);
    `UNUSED_VAR ({vld_mask, cls_c})

    // Intermediate Signals
    wire [TCK-1:0] nan_in_tf32, inf_z_tf32, inf_op_tf32, sign_tf32;
    wire [TCK-1:0] nan_in_fp16, inf_z_fp16, inf_op_fp16, sign_fp16;
    wire [TCK-1:0] nan_in_bf16, inf_z_bf16, inf_op_bf16, sign_bf16;

    // ----------------------------------------------------------------------
    // 1. TF32 Preparation (Even lanes only)
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_tf32
        if ((i % 2) == 0) begin
            fedp_class_t ca = cls_tf32[0][i/2];
            fedp_class_t cb = cls_tf32[1][i/2];
            `UNUSED_VAR ({ca, cb})
            assign nan_in_tf32[i] = ca.is_nan | cb.is_nan;
            assign inf_z_tf32[i]  = (ca.is_inf & cb.is_zero) | (ca.is_zero & cb.is_inf);
            assign inf_op_tf32[i] = ca.is_inf | cb.is_inf;
            assign sign_tf32[i]   = ca.sign ^ cb.sign;
        end else begin
            assign nan_in_tf32[i] = 1'b0;
            assign inf_z_tf32[i]  = 1'b0;
            assign inf_op_tf32[i] = 1'b0;
            assign sign_tf32[i]   = 1'b0;
        end
    end

    // ----------------------------------------------------------------------
    // 2. FP16 Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_fp16
        fedp_class_t ca = cls_fp16[0][i];
        fedp_class_t cb = cls_fp16[1][i];
        `UNUSED_VAR ({ca, cb})
        assign nan_in_fp16[i] = ca.is_nan | cb.is_nan;
        assign inf_z_fp16[i]  = (ca.is_inf & cb.is_zero) | (ca.is_zero & cb.is_inf);
        assign inf_op_fp16[i] = ca.is_inf | cb.is_inf;
        assign sign_fp16[i]   = ca.sign ^ cb.sign;
    end

    // ----------------------------------------------------------------------
    // 3. BF16 Preparation
    // ----------------------------------------------------------------------
    for (genvar i = 0; i < TCK; ++i) begin : g_prep_bf16
        fedp_class_t ca = cls_bf16[0][i];
        fedp_class_t cb = cls_bf16[1][i];
        `UNUSED_VAR ({ca, cb})
        assign nan_in_bf16[i] = ca.is_nan | cb.is_nan;
        assign inf_z_bf16[i]  = (ca.is_inf & cb.is_zero) | (ca.is_zero & cb.is_inf);
        assign inf_op_bf16[i] = ca.is_inf | cb.is_inf;
        assign sign_bf16[i]   = ca.sign ^ cb.sign;
    end

    // ----------------------------------------------------------------------
    // 4. Merge & Mask
    // ----------------------------------------------------------------------
    wire [TCK-1:0] prod_nan, prod_inf, prod_sign;

    for (genvar i = 0; i < TCK; ++i) begin : g_merge
        logic n_in, i_z, i_op, sgn, valid_lane;

        always_comb begin
            case (fmtf)
                TCU_FP32_ID: begin
                    n_in = nan_in_tf32[i]; i_z = inf_z_tf32[i]; i_op = inf_op_tf32[i]; sgn = sign_tf32[i];
                    valid_lane = ((i % 2) == 0) ? vld_mask[i * 4] : 1'b0;
                end
                TCU_FP16_ID: begin
                    n_in = nan_in_fp16[i]; i_z = inf_z_fp16[i]; i_op = inf_op_fp16[i]; sgn = sign_fp16[i];
                    valid_lane = vld_mask[i * 4];
                end
                TCU_BF16_ID: begin
                    n_in = nan_in_bf16[i]; i_z = inf_z_bf16[i]; i_op = inf_op_bf16[i]; sgn = sign_bf16[i];
                    valid_lane = vld_mask[i * 4];
                end
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
    // 5. Global Aggregation
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
