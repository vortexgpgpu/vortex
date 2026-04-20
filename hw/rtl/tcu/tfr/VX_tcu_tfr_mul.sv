// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0
// See the License for the specific language governing permissions and limitations.
`include "VX_define.vh"

module VX_tcu_tfr_mul import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,            // Number of 32-bit input registers
    parameter W = 25,           // Product width
    parameter WA = 28,          // Accumulator width
    parameter EXP_W = 10,       // Max exponent width
    parameter TCK = 2 * N,      // Max physical lanes
    parameter LATENCY = 1       // MUL stage latency (1 or 2)
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire              valid_in,
    input wire [31:0]       req_id,

    input wire [3:0]        fmt_s,

    input wire [N-1:0][31:0] a_row,
    input wire [N-1:0][31:0] b_col,
    input wire [31:0]       c_val,
    input wire [7:0]        sf_a,
    input wire [7:0]        sf_b,
    input wire [TCU_MAX_INPUTS-1:0] vld_mask,

    // Outputs
    output wire [EXP_W-1:0]         max_exp,
    output wire [TCK:0][7:0] shift_amts,
    output wire [TCK:0][W-1:0] raw_sigs,
    output fedp_excep_t              exception,
    output wire [TCK-1:0]     lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    localparam EXC_W = $bits(fedp_excep_t);

    // Input Fanout Replication
    (* keep = "true" *) wire [N-1:0][31:0] a_row_f16 = a_row;
    (* keep = "true" *) wire [N-1:0][31:0] b_col_f16 = b_col;
    (* keep = "true" *) wire [N-1:0][31:0] a_row_f8  = a_row;
    (* keep = "true" *) wire [N-1:0][31:0] b_col_f8  = b_col;
    (* keep = "true" *) wire [N-1:0][31:0] a_row_int = a_row;
    (* keep = "true" *) wire [N-1:0][31:0] b_col_int = b_col;

    // ======================================================================
    // 1. Independent Compute Paths
    // ======================================================================

    // --- F16 / BF16 / TF32 ------------------------------------------------
    wire [TCK-1:0][W-1:0]     mul_f16_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f16_exp;
    fedp_excep_t [TCK-1:0]    mul_f16_exc;

    fedp_class_t [2*TCK-1:0] cls_fp8 [2];
    VX_tcu_tfr_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_FP8_ID)) c_a_fp8 (.val(a_row), .cls(cls_fp8[0]));
    VX_tcu_tfr_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_FP8_ID)) c_b_fp8 (.val(b_col), .cls(cls_fp8[1]));

    fedp_class_t [2*TCK-1:0] cls_bf8 [2];
    VX_tcu_tfr_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_BF8_ID)) c_a_bf8 (.val(a_row), .cls(cls_bf8[0]));
    VX_tcu_tfr_classifier #(.N(4 * N), .WIDTH(8), .FMT(TCU_BF8_ID)) c_b_bf8 (.val(b_col), .cls(cls_bf8[1]));

    fedp_class_t [0:0] cls_c;
    VX_tcu_tfr_classifier #(.N(1), .WIDTH(32), .FMT(TCU_FP32_ID)) c_c (.val(c_val), .cls(cls_c));

    // ----------------------------------------------------------------------
    // 2. Mantissa Product
    // ----------------------------------------------------------------------

    wire [TCK-1:0][5:0] exp_diff_f8;
    wire [TCK:0][EXP_W-1:0] raw_exps;

    VX_tcu_tfr_shared_mul #(
        .N   (N),
        .TCK (TCK)
    ) shared_mul_inst (
        .vld_mask   (vld_mask),
        .fmt_s      (fmt_s),
        .a_row      (a_row),
        .b_col      (b_col),
        .c_val      (c_val),
        .sf_a       (sf_a),
        .sf_b       (sf_b),
        .cls_tf32   (cls_tf32),
        .cls_fp16   (cls_fp16),
        .cls_bf16   (cls_bf16),
        .cls_fp8    (cls_fp8),
        .cls_bf8    (cls_bf8),
        .cls_c      (cls_c),
        .exp_diff_f8(exp_diff_f8),
        .y          (raw_sigs)
    );

    // ----------------------------------------------------------------------
    // 3. Exponent Product
    // ----------------------------------------------------------------------

    VX_tcu_tfr_exp_bias #(
        .N   (N),
        .TCK (TCK),
        .W   (W),
        .WA  (WA),
        .EXP_W(EXP_W)
    ) mul_f16 (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask       (vld_mask),
        .fmt_f      (fmt_s[2:0]),
        .a_row      (a_row_f16),
        .b_col      (b_col_f16),
        .result_sig (mul_f16_sig),
        .result_exp (mul_f16_exp),
        .exceptions (mul_f16_exc)
    );

    // --- FP8 / BF8 --------------------------------------------------------
    wire [TCK-1:0][W-1:0]     mul_f8_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f8_exp;
    fedp_excep_t [TCK-1:0]    mul_f8_exc;

    VX_tcu_drl_mul_f8 #(
        .N(N),
        .TCK(TCK),
        .W(W),
        .WA(WA),
        .EXP_W(EXP_W)
    ) mul_f8 (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_f      (fmt_s[2:0]),
        .a_row      (a_row_f8),
        .b_col      (b_col_f8),
        .result_sig (mul_f8_sig),
        .result_exp (mul_f8_exp),
        .exceptions (mul_f8_exc)
    );

    // --- I8/U8/I4/U4 ------------------------------------------------------
    wire [TCK-1:0][W-1:0] mul_int_sig;
    VX_tcu_tfr_mul_int #(
        .N(N),
        .TCK(TCK)
    ) mul_int (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[2:0]),
        .a_row      (a_row_int),
        .b_col      (b_col_int),
        .sf_a       (sf_a),
        .sf_b       (sf_b),
        .result     (mul_int_sig)
    );

    // ======================================================================
    // 2. Aggregation
    // ======================================================================

    wire [TCK:0][EXP_W-1:0]  join_exponents;
    wire [TCK:0][W-1:0]      join_raw_sigs;
    fedp_excep_t [TCK:0]     join_exceptions;

    VX_tcu_tfr_mul_join #(
        .N(N),
        .TCK(TCK),
        .W(W),
        .WA(WA),
        .EXP_W(EXP_W)
    ) join_stage (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),

        .fmt_s      (fmt_s),
        .c_val      (c_val),

        .sig_f16    (mul_f16_sig),
        .exp_f16    (mul_f16_exp),
        .exc_f16    (mul_f16_exc),

        .sig_f8     (mul_f8_sig),
        .exp_f8     (mul_f8_exp),
        .exc_f8     (mul_f8_exc),

         .sig_int   (mul_int_sig),

        .sig_out    (join_raw_sigs),
        .exp_out    (join_exponents),
        .exc_out    (join_exceptions)
    );

    wire [TCK-1:0] join_lane_mask;

    VX_tcu_tfr_exceptions #(
        .N   (N),
        .TCK (TCK)
    ) exceptions_inst (
        .vld_mask   (vld_mask),
        .fmtf       (fmt_s[2:0]),
        .cls_tf32   (cls_tf32),
        .cls_fp16   (cls_fp16),
        .cls_bf16   (cls_bf16),
        .cls_fp8    (cls_fp8),
        .cls_bf8    (cls_bf8),
        .cls_c      (cls_c),
        .exceptions (exceptions)
    );

    // ----------------------------------------------------------------------
    // 6. Lane Mask
    // ----------------------------------------------------------------------
    VX_tcu_tfr_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask (vld_mask),
        .fmt_s    (fmt_s),
        .lane_mask(join_lane_mask)
    );

    // ======================================================================
    // 3. Exception Reduction & Max Exponent
    // ======================================================================

    wire [TCK:0][EXP_W-1:0] r_exponents;
    wire [TCK:0][W-1:0]     r_raw_sigs;
    fedp_excep_t [TCK:0]    r_exceptions;
    wire [TCK-1:0]          r_lane_mask;

    VX_pipe_register #(
        .DATAW ((TCK+1)*EXP_W + (TCK+1)*W + (TCK+1)*EXC_W + TCK),
        .DEPTH (LATENCY - 1)
    ) pipe_join (
        .clk     (clk),
        .reset   (reset),
        .enable  (enable),
        .data_in ({join_exponents, join_raw_sigs, join_exceptions, join_lane_mask}),
        .data_out({r_exponents,    r_raw_sigs,    r_exceptions,    r_lane_mask})
    );

    VX_tcu_tfr_exc_reduce #(
        .TCK (TCK)
    ) exc_reduce (
        .exc_in  (r_exceptions),
        .exc_out (exception)
    );

    VX_tcu_tfr_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_max_exp (
        .exponents (r_exponents),
        .max_exp    (max_exp),
        .shift_amts (shift_amts)
    );

    // ======================================================================
    // 4. Outputs
    // ======================================================================

    assign raw_sigs  = r_raw_sigs;
    assign lane_mask = r_lane_mask;

endmodule
