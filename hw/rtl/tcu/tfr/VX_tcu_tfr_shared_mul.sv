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

module VX_tcu_tfr_shared_mul import VX_tcu_pkg::*;  #(
    parameter `STRING INSTANCE_ID = "",
    parameter N = 2,            // Number of 32-bit input registers
    parameter W = 25,           // Product width
    parameter WA = 28,          // Accumulator width
    parameter EXP_W = 10,       // Max exponent width
    parameter TCK = 2 * N,      // Max physical lanes
    parameter SF = 1            // Scale factor slots
) (
    input wire              clk,
    input wire              valid_in,
    input wire [31:0]       req_id,

    input wire [TCU_MAX_INPUTS-1:0] vld_mask,

    input wire [4:0]        fmt_s,

    input wire [N-1:0][31:0] a_row,
    input wire [N-1:0][31:0] b_col,
    input wire [31:0]        c_val,
`ifdef TCU_MX_ENABLE
    input wire [SF-1:0][7:0] sf_a,
    input wire [SF-1:0][7:0] sf_b,
`endif
    output wire [TCK:0][EXP_W-1:0] exponents,
    output wire [TCK:0]            exp_sel,
    output wire [TCK-1:0][TCK-1:0][EXP_W:0] exp_diff_mat,

    output wire [TCK:0][24:0] raw_sigs,
    output wire fedp_excep_t  exceptions,
    output wire [TCK-1:0]     lane_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifndef TCU_MX_ENABLE
    `UNUSED_PARAM (SF)
`endif
    `UNUSED_VAR ({clk, req_id, valid_in})

`ifdef TCU_FP16_ENABLE
`define TFR_MUL_F16_ENABLE
`elsif TCU_TF32_ENABLE
`define TFR_MUL_F16_ENABLE
`endif


`ifdef TFR_MUL_F16_ENABLE
    // F16 / BF16 / TF32
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
        .fmt_f      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result_sig (mul_f16_sig),
        .result_exp (mul_f16_exp),
        .exceptions (mul_f16_exc)
    );
`endif

`ifdef TCU_FP8_ENABLE
    // FP8 / BF8 / MXFP8 / MXBF8
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
        .fmt_f      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
    `ifdef TCU_MX_ENABLE
        .sf_a       (sf_a[0]),
        .sf_b       (sf_b[0]),
    `endif
        .result_sig (mul_f8_sig),
        .result_exp (mul_f8_exp),
        .exceptions (mul_f8_exc)
    );
`endif

`ifdef TCU_MX_ENABLE
`ifdef TCU_FP4_ENABLE
    // MXFP4 / NVFP4
    wire [TCK-1:0][24:0]      mul_f4_sig;
    wire [TCK-1:0][EXP_W-1:0] mul_f4_exp;
    fedp_excep_t [TCK-1:0]    mul_f4_exc;

    wire [SF-1:0][TCK-1:0][24:0]      mul_f4_sig_s;
    wire [SF-1:0][TCK-1:0][EXP_W-1:0] mul_f4_exp_s;
    fedp_excep_t [SF-1:0][TCK-1:0]    mul_f4_exc_s;
    localparam SF_IDX_W = (SF > 1) ? $clog2(SF) : 1;

    for (genvar s = 0; s < SF; ++s) begin : g_mul_f4_sf
        VX_tcu_tfr_mul_f4 #(
            .N(N),
            .TCK(TCK),
            .W(W),
            .WA(WA),
            .EXP_W(EXP_W)
        ) mul_f4 (
            .clk        (clk),
            .valid_in   (valid_in),
            .req_id     (req_id),
            .vld_mask   (vld_mask),
            .fmt_f      (fmt_s[3:0]),
            .a_row      (a_row),
            .b_col      (b_col),
            .sf_a       (sf_a[s]),
            .sf_b       (sf_b[s]),
            .result_sig (mul_f4_sig_s[s]),
            .result_exp (mul_f4_exp_s[s]),
            .exceptions (mul_f4_exc_s[s])
        );
    end

    for (genvar i = 0; i < TCK; ++i) begin : g_mul_f4_lane
        localparam K_WORD = i / 2;
        wire is_4_bit_block16 = (fmt_s == TCU_NVFP4_ID);
        wire [SF_IDX_W-1:0] mx_slot_4b = is_4_bit_block16
            ? SF_IDX_W'(((K_WORD / 2) < SF) ? (K_WORD / 2) : (SF - 1))
            : SF_IDX_W'(((K_WORD / 4) < SF) ? (K_WORD / 4) : (SF - 1));
        assign mul_f4_sig[i] = mul_f4_sig_s[mx_slot_4b][i];
        assign mul_f4_exp[i] = mul_f4_exp_s[mx_slot_4b][i];
        assign mul_f4_exc[i] = mul_f4_exc_s[mx_slot_4b][i];
    end
`endif
`endif

`ifdef TCU_INT8_ENABLE
    // I8 / U8 / MXI8
    wire [TCK-1:0][24:0] mul_int8_sig;
    VX_tcu_tfr_mul_i8 #(
        .N(N),
        .TCK(TCK)
    ) mul_int8 (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
    `ifdef TCU_MX_ENABLE
        .sf_a       (sf_a[0]),
        .sf_b       (sf_b[0]),
    `endif
        .result     (mul_int8_sig)
    );
`endif

`ifdef TCU_INT4_ENABLE
    // I4 / U4
    wire [TCK-1:0][24:0] mul_int4_sig;
    VX_tcu_tfr_mul_i4 #(
        .N(N),
        .TCK(TCK)
    ) mul_int4 (
        .clk        (clk),
        .valid_in   (valid_in),
        .req_id     (req_id),
        .vld_mask   (vld_mask),
        .fmt_i      (fmt_s[3:0]),
        .a_row      (a_row),
        .b_col      (b_col),
        .result     (mul_int4_sig)
    );
`endif

    // Aggregation and exception reduction
    fedp_excep_t [TCK:0] join_exceptions;

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

    `ifdef TFR_MUL_F16_ENABLE
        .sig_f16    (mul_f16_sig),
        .exp_f16    (mul_f16_exp),
        .exc_f16    (mul_f16_exc),
    `endif

    `ifdef TCU_FP8_ENABLE
        .sig_f8     (mul_f8_sig),
        .exp_f8     (mul_f8_exp),
        .exc_f8     (mul_f8_exc),
    `endif

    `ifdef TCU_MX_ENABLE
    `ifdef TCU_FP4_ENABLE
        .sig_f4     (mul_f4_sig),
        .exp_f4     (mul_f4_exp),
        .exc_f4     (mul_f4_exc),
    `endif
    `endif

    `ifdef TCU_INT8_ENABLE
        .sig_int8   (mul_int8_sig),
    `endif
    `ifdef TCU_INT4_ENABLE
        .sig_int4   (mul_int4_sig),
    `endif

        .sig_out    (raw_sigs),
        .exp_out    (exponents),
        .exc_out    (join_exceptions)
    );

    VX_tcu_tfr_exc_reduce #(
        .TCK (TCK)
    ) exc_reduce (
        .exc_in  (join_exceptions),
        .exc_out (exceptions)
    );

    // Maximum exponent index + difference matrix
    VX_tcu_tfr_max_exp #(
        .N     (TCK+1),
        .WIDTH (EXP_W)
    ) find_diff_mat (
        .exponents (exponents),
        .sel_exp   (exp_sel),
        .diff_mat  (exp_diff_mat)
    );

    // Lane mask
    VX_tcu_tfr_lane_mask #(
        .N   (N),
        .TCK (TCK)
    ) lane_mask_inst (
        .vld_mask (vld_mask),
        .fmt_s    (fmt_s),
        .lane_mask(lane_mask)
    );

endmodule

`ifdef TFR_MUL_F16_ENABLE
`undef TFR_MUL_F16_ENABLE
`endif
