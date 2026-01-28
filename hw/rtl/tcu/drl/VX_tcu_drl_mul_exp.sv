`include "VX_define.vh"

module VX_tcu_drl_mul_exp import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,            // Number of 32-bit input registers
    parameter W = 25,           // Accumulator/Mantissa Width
    parameter EXP_W = 10,
    parameter TCK = 2 * N       // Max physical lanes
) (
    input wire              clk,
    input wire              valid_in,
    input wire [31:0]       req_id,
    input wire [3:0]        fmt_s,

    input wire [N-1:0][31:0] a_row,
    input wire [N-1:0][31:0] b_col,
    input wire [31:0]       c_val,
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,

    output wire [9:0]         max_exp,
    output wire [TCK:0][7:0]  shift_amt,
    output wire [TCK:0][24:0] raw_sigs,
    output wire fedp_excep_t  exceptions,
    output wire [TCK-1:0]     lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    // ----------------------------------------------------------------------
    // 1. Classification
    // ----------------------------------------------------------------------
    fedp_class_t [N-1:0] cls_tf32 [2];
    VX_tcu_drl_classifier #(.N(N), .WIDTH(32), .FMT(TCU_FP32_ID)) c_a_tf32 (.val(a_row), .cls(cls_tf32[0]));
    VX_tcu_drl_classifier #(.N(N), .WIDTH(32), .FMT(TCU_FP32_ID)) c_b_tf32 (.val(b_col), .cls(cls_tf32[1]));

    fedp_class_t [TCK-1:0] cls_fp16 [2];
    VX_tcu_drl_classifier #(.N(2 * N), .WIDTH(16), .FMT(TCU_FP16_ID)) c_a_fp16 (.val(a_row), .cls(cls_fp16[0]));
    VX_tcu_drl_classifier #(.N(2 * N), .WIDTH(16), .FMT(TCU_FP16_ID)) c_b_fp16 (.val(b_col), .cls(cls_fp16[1]));

    fedp_class_t [TCK-1:0] cls_bf16 [2];
    VX_tcu_drl_classifier #(.N(2 * N), .WIDTH(16), .FMT(TCU_BF16_ID)) c_a_bf16 (.val(a_row), .cls(cls_bf16[0]));
    VX_tcu_drl_classifier #(.N(2 * N), .WIDTH(16), .FMT(TCU_BF16_ID)) c_b_bf16 (.val(b_col), .cls(cls_bf16[1]));

    fedp_class_t [2*TCK-1:0] cls_fp8 [2];
    VX_tcu_drl_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_FP8_ID)) c_a_fp8 (.val(a_row), .cls(cls_fp8[0]));
    VX_tcu_drl_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_FP8_ID)) c_b_fp8 (.val(b_col), .cls(cls_fp8[1]));

    fedp_class_t [2*TCK-1:0] cls_bf8 [2];
    VX_tcu_drl_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_BF8_ID)) c_a_bf8 (.val(a_row), .cls(cls_bf8[0]));
    VX_tcu_drl_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_BF8_ID)) c_b_bf8 (.val(b_col), .cls(cls_bf8[1]));

    fedp_class_t [0:0] cls_c;
    VX_tcu_drl_classifier #(.N(1), .WIDTH(32), .FMT(TCU_FP32_ID)) c_c (.val(c_val), .cls(cls_c));

    // ----------------------------------------------------------------------
    // 2. Mantissa Product
    // ----------------------------------------------------------------------

    wire [TCK-1:0]      exp_low_larger;
    wire [TCK-1:0][6:0] raw_exp_diff;
    wire [TCK:0][9:0]   raw_exps;

    VX_tcu_drl_shared_mul #(
        .N   (N),
        .TCK (TCK)
    ) shared_mul_inst (
        .vld_mask       (vld_mask),
        .fmt_s          (fmt_s),
        .a_row          (a_row),
        .b_col          (b_col),
        .c_val          (c_val),
        .cls_tf32       (cls_tf32),
        .cls_fp16       (cls_fp16),
        .cls_bf16       (cls_bf16),
        .cls_fp8        (cls_fp8),
        .cls_bf8        (cls_bf8),
        .cls_c          (cls_c[0]),
        .exp_low_larger (exp_low_larger),
        .raw_exp_diff   (raw_exp_diff),
        .y              (raw_sigs)
    );

    // ----------------------------------------------------------------------
    // 3. Exponent Product
    // ----------------------------------------------------------------------

    VX_tcu_drl_exp_bias #(
        .N   (N),
        .TCK (TCK),
        .W   (W),
        .EXP_W(EXP_W)
    ) exp_bias_inst (
        .vld_mask       (vld_mask),
        .fmtf           (fmt_s[2:0]),
        .a_row          (a_row),
        .b_col          (b_col),
        .c_val          (c_val),
        .cls_tf32       (cls_tf32),
        .cls_fp16       (cls_fp16),
        .cls_bf16       (cls_bf16),
        .cls_fp8        (cls_fp8),
        .cls_bf8        (cls_bf8),
        .cls_c          (cls_c[0]),
        .raw_exp_y      (raw_exps),
        .exp_low_larger (exp_low_larger),
        .raw_exp_diff   (raw_exp_diff)
    );

    // ----------------------------------------------------------------------
    // 4. Max Exponent
    // ----------------------------------------------------------------------

    VX_tcu_drl_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_max_exp (
        .exponents (raw_exps),
        .max_exp   (max_exp),
        .shift_amt (shift_amt)
    );

    // ----------------------------------------------------------------------
    // 5. Exception Flags
    // ----------------------------------------------------------------------

    VX_tcu_drl_exceptions #(
        .N   (N),
        .TCK (TCK)
    ) exceptions_inst (
        .vld_mask   (vld_mask),
        .fmtf       (fmt_s[2:0]),
        .cls_tf32   (cls_tf32),
        .cls_fp16   (cls_fp16),
        .cls_bf16   (cls_bf16),
        .cls_c      (cls_c[0]),
        .exceptions (exceptions)
    );

    // ----------------------------------------------------------------------
    // 6. Lane Mask
    // ----------------------------------------------------------------------
    VX_tcu_drl_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask (vld_mask),
        .fmt_s    (fmt_s),
        .lane_mask(lane_mask)
    );

endmodule
