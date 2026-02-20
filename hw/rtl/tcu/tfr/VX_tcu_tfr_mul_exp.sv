// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0
// See the License for the specific language governing permissions and limitations.
`include "VX_define.vh"

module VX_tcu_tfr_mul_exp import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,            // Number of 32-bit input registers
    parameter W = 25,           // Product width
    parameter WA = 28,          // Accumulator width
    parameter EXP_W = 10,       // Max exponent width
    parameter TCK = 2 * N       // Max physical lanes
) (
    input wire              clk,
    input wire              valid_in,
    input wire [31:0]       req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,

    input wire [3:0]        fmt_s,

    input wire [N-1:0][31:0] a_row,
    input wire [N-1:0][31:0] b_col,
    input wire [31:0]        c_val,
    input wire [7:0]         sf_a,
    input wire [7:0]         sf_b,

    output wire [EXP_W-1:0]   max_exp,
    output wire [TCK:0][EXP_W-1:0] exponents,

    output wire [TCK:0][24:0] raw_sigs,
    output wire fedp_excep_t  exceptions,
    output wire [TCK-1:0]     lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})

    // ======================================================================
    // 1. Independent Compute Paths
    // ======================================================================

    // --- F16 / BF16 / TF32 ------------------------------------------------
    wire [TCK-1:0][24:0]      mul_f16_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f16_exp;
    fedp_excep_t [TCK-1:0]    mul_f16_exc;

    VX_tcu_tfr_mul_f16 #(
        .N(N),
        .TCK(TCK),
        .W(W),
        .WA(WA),
        .EXP_W(EXP_W)
    ) mul_f16 (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_f      (fmt_s[2:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result_sig (mul_f16_sig),
        .result_exp (mul_f16_exp),
        .exceptions (mul_f16_exc)
    );

    // --- BF8 / BF8 --------------------------------------------------------
    wire [TCK-1:0][24:0]      mul_f8_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f8_exp;
    fedp_excep_t [TCK-1:0]    mul_f8_exc;

    VX_tcu_tfr_mul_f8 #(
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
        .a_row      (a_row),
        .b_col      (b_col),
        .result_sig (mul_f8_sig),
        .result_exp (mul_f8_exp),
        .exceptions (mul_f8_exc)
    );

    // --- I8/U8/I4/U4 ------------------------------------------------------
    wire [TCK-1:0][24:0] mul_int_sig;
    VX_tcu_tfr_mul_int #(
        .N(N),
        .TCK(TCK)
    ) mul_int (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[2:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .sf_a       (sf_a),
        .sf_b       (sf_b),
        .result     (mul_int_sig)
    );

    // ======================================================================
    // 2. Aggregation & Exception Reduction
    // ======================================================================

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

        .sig_out    (raw_sigs),
        .exp_out    (exponents),
        .exc_out    (exceptions)
    );

    // ======================================================================
    // 3. Global Maximum Exponent
    // ======================================================================

    VX_tcu_tfr_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_max_exp (
        .exponents (exponents),
        .max_exp   (max_exp)
    );

    // ======================================================================
    // 4. Lane Mask
    // ======================================================================

    VX_tcu_tfr_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask (vld_mask),
        .fmt_s    (fmt_s),
        .lane_mask(lane_mask)
    );

endmodule
