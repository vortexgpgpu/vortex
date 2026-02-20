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

module VX_tcu_tfr_mul_join import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 2,
    parameter TCK   = 2 * N,
    parameter W     = 25,
    parameter WA    = 28,
    parameter EXP_W = 10
) (
    input wire                      clk,
    input wire                      valid_in,
    input wire [31:0]               req_id,

    input wire [3:0]                fmt_s,
    input wire [31:0]               c_val,

    // Inputs from the Split Paths
    input wire [TCK-1:0][24:0]      sig_f16,
    input wire [TCK-1:0][EXP_W-1:0] exp_f16,
    input fedp_excep_t [TCK-1:0]    exc_f16,

    input wire [TCK-1:0][24:0]      sig_f8,
    input wire [TCK-1:0][EXP_W-1:0] exp_f8,
    input fedp_excep_t [TCK-1:0]    exc_f8,
    input wire [TCK-1:0][24:0]      sig_int,

    output logic [TCK:0][24:0]      sig_out,
    output logic [TCK:0][EXP_W-1:0] exp_out,
    output fedp_excep_t             exc_out
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in})
    `UNUSED_VAR ({sig_f8, exp_f8, exc_f8})
    `UNUSED_VAR ({sig_int})


    // ======================================================================
    // 1. Path Selection (Muxing)
    // ======================================================================

    logic [TCK-1:0][24:0]      sig_sel;
    logic [TCK-1:0][EXP_W-1:0] exp_sel;
    fedp_excep_t [TCK-1:0]     exc_sel;

    always_comb begin
        case (fmt_s)
            // --- F16 / BF16 / TF32 ---
        `ifdef TCU_BF16_ENABLE
            TCU_BF16_ID,
        `endif
        `ifdef TCU_TF32_ENABLE
            TCU_TF32_ID,
        `endif
            TCU_FP16_ID: begin
                sig_sel = sig_f16;
                exp_sel = exp_f16;
                exc_sel = exc_f16;
            end

            // --- FP8 / BF8 ---
        `ifdef TCU_FP8_ENABLE
            TCU_FP8_ID, TCU_BF8_ID: begin
                sig_sel = sig_f8;
                exp_sel = exp_f8;
                exc_sel = exc_f8;
            end
        `endif

            // --- Integers ---
        `ifdef TCU_INT_ENABLE
            TCU_I8_ID, TCU_U8_ID, TCU_I4_ID, TCU_U4_ID, TCU_MXI8_ID: begin
                sig_sel = sig_int;
                exp_sel = 'x;
                exc_sel = 'x;
            end
        `endif
            default: begin
                sig_sel = 'x;
                exp_sel = 'x;
                exc_sel = 'x;
            end
        endcase
    end

    wire c_is_int = tcu_fmt_is_int(fmt_s);

    // ======================================================================
    // 2. C-Term Handling
    // ======================================================================

    // 2a. Local Classifier for C (Check for Zero/Nan/Inf)
    fedp_class_t cls_c;
    VX_tcu_tfr_classifier #(.N(1), .WIDTH(32), .FMT(TCU_FP32_ID)) c_c (.val(c_val), .cls(cls_c));
    `UNUSED_VAR (cls_c)

    // 2b. C-Term Signal Extraction
    // If format is INT, we just take the bits. If float, we add hidden bit.
    wire [24:0] c_sig_final = c_is_int ? c_val[24:0] : {c_val[31], 1'b1, c_val[22:0]};

    // 2c. C-Term Exponent Calculation
    wire [EXP_W-1:0] c_exp_adj = EXP_W'(c_val[30:23]) - EXP_W'(W-1) + EXP_W'(WA-1) + 128;
    wire [EXP_W-1:0] c_exp_final = cls_c.is_zero ? '0 : c_exp_adj;

    // ======================================================================
    // 3. Output Aggregation (Signal & Exponent)
    // ======================================================================

    // Combine Lanes + C-Term
    // Index TCK is C-term, 0..TCK-1 are lanes
    assign sig_out = {c_sig_final, sig_sel};
    assign exp_out = {c_exp_final, exp_sel};

    // ======================================================================
    // 4. Exception Reduction (Lanes + C-term)
    // ======================================================================

    // Unpack lane exceptions for vector operations
    logic [TCK-1:0] lane_nan;
    logic [TCK-1:0] lane_inf;
    logic [TCK-1:0] lane_sign;

    for (genvar i = 0; i < TCK; ++i) begin : g_unpack_exc
        assign lane_nan[i]  = exc_sel[i].is_nan;
        assign lane_inf[i]  = exc_sel[i].is_inf;
        assign lane_sign[i] = exc_sel[i].sign;
    end

    // 4a. C-Term flags
    // If Integer mode, C-term exceptions are suppressed (always valid)
    wire c_is_nan = cls_c.is_nan;
    wire c_is_inf = cls_c.is_inf;
    wire c_sign   = cls_c.sign; // Relevant only if Inf

    // 4b. Global Infinity Analysis
    // We must detect if we are adding +Inf and -Inf anywhere in the dot product.
    // This includes (Lane vs Lane) AND (Lanes vs C-term).

    wire [TCK-1:0] p_pos_inf = lane_inf & ~lane_sign;
    wire [TCK-1:0] p_neg_inf = lane_inf &  lane_sign;

    wire c_pos_inf = c_is_inf & ~c_sign;
    wire c_neg_inf = c_is_inf &  c_sign;

    // Check if we have ANY positive infinity (Lanes or C)
    wire has_pos = (|p_pos_inf) | c_pos_inf;

    // Check if we have ANY negative infinity (Lanes or C)
    wire has_neg = (|p_neg_inf) | c_neg_inf;

    // 4c. Final Logic
    // NaN if:
    // 1. Any input (Lane product or C-term) is NaN
    // 2. We have both +Inf and -Inf in the sum (Invalid Operation)
    wire any_input_nan = (|lane_nan) | c_is_nan;
    wire add_gen_nan   = has_pos & has_neg;
    wire res_nan       = any_input_nan | add_gen_nan;

    // Inf if:
    // 1. We have an Inf (Pos or Neg)
    // 2. Result is NOT NaN
    wire res_inf = (has_pos | has_neg) & ~res_nan;

    // Sign:
    // If result is Inf, sign is 1 if we have NegInf (and no PosInf, implied by !NaN).
    // If result is Normal/Zero, sign is determined by the final adder (Stage 3),
    // but typically the exception structure reports the "Special Case" sign here.
    wire res_sign = has_neg & ~has_pos;

    assign exc_out.is_nan = res_nan;
    assign exc_out.is_inf = res_inf;
    assign exc_out.sign   = res_sign;

endmodule
