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

module VX_tcu_tfr_norm_round import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WA = 30,
    parameter EXP_W = 10,
    parameter C_HI_W = 8
) (
    input wire          clk,
    input wire          valid_in,
    input wire [31:0]   req_id,
    input wire [EXP_W-1:0] max_exp,
    input wire [WA-1:0] acc_sig,   // Sum magnitude
    input wire          acc_sign,  // Sum sign
    input wire [C_HI_W-1:0] cval_hi,
    input wire          is_int,
    input wire          sticky_in,
    input fedp_excep_t  exceptions,
    output wire [31:0]  result
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, req_id, valid_in, is_int, cval_hi})

    // The accumulator resolves the sum in sign-magnitude form, so the leading
    // zero count runs directly on the exact magnitude: no abs conversion, no
    // prediction slack, and the normalized MSB always lands on bit WA-1.
    wire sum_sign = acc_sign;
    wire [WA-1:0] abs_sum = acc_sig;
    wire zero_sum = ~|acc_sig;

    wire [$clog2(WA)-1:0] lz_count;
    VX_lzc #(
        .N(WA)
    ) lzc_inst (
        .data_in   (abs_sum),
        .data_out  (lz_count),
        `UNUSED_PIN (valid_out)
    );

    // Parallel exponent calculation: norm_exp_base = max_exp - (lz_count + 128)
    wire signed [EXP_W-1:0] norm_exp_base;
    wire signed [EXP_W-1:0] norm_exp_plus1;

    wire [EXP_W-1:0] sub_term = EXP_W'({1'b1, 2'b00, lz_count});
    VX_ks_adder #(
        .N(EXP_W),
        .BYPASS (`FORCE_BUILTIN_ADDER(EXP_W))
    ) exp_sub (
        .cin   (1'b1),
        .dataa (max_exp),
        .datab (~sub_term),
        .sum   (norm_exp_base),
        `UNUSED_PIN (cout)
    );

    assign norm_exp_plus1 = norm_exp_base + EXP_W'(1'b1);

    // Normalization shift: exact count, fixed window
    wire [WA-1:0] shifted_sum = abs_sum << lz_count;

    wire [26:0] aligned_bits = shifted_sum[WA-1 -: 27];

    // Parallel rounding (select-add: the incremented mantissa is computed
    // alongside the round decision, a mux picks the outcome)
    wire [23:0] norm_man   = aligned_bits[26:3];
    wire        guard_bit  = aligned_bits[2];
    wire        round_bit  = aligned_bits[1];
    wire        sticky_bit = aligned_bits[0] | (|shifted_sum[WA-27:0]) | sticky_in;
    wire        lsb_bit    = norm_man[0];
    wire round_up = guard_bit && (round_bit || sticky_bit || lsb_bit);

    wire [24:0] man_plus_zero = {1'b0, norm_man};
    wire [24:0] man_plus_one;
    VX_ks_adder #(
        .N(25),
        .BYPASS (`FORCE_BUILTIN_ADDER(25))
    ) round_adder (
        .cin   (1'b1),
        .dataa (man_plus_zero),
        .datab (25'd0),
        .sum   (man_plus_one),
        `UNUSED_PIN (cout)
    );

    wire [24:0] rounded_sig_full = round_up ? man_plus_one : man_plus_zero;
    wire carry_out = rounded_sig_full[24];
    wire [22:0] final_man = carry_out ? rounded_sig_full[23:1] : rounded_sig_full[22:0];

    // Final exponent and exception
    wire signed [EXP_W-1:0] final_exp_s = carry_out ? norm_exp_plus1 : norm_exp_base;

    logic [7:0] packed_exp;
    logic exp_overflow, exp_underflow;
    always_comb begin
        if (final_exp_s >= 255) begin
            packed_exp = 8'hFF;
            exp_overflow = 1'b1;
            exp_underflow = 1'b0;
        end else if (final_exp_s <= 0) begin
            packed_exp = 8'h00;
            exp_overflow = 1'b0;
            exp_underflow = 1'b1;
        end else begin
            packed_exp = final_exp_s[7:0];
            exp_overflow = 1'b0;
            exp_underflow = 1'b0;
        end
    end

    wire [31:0] fp_nan_result = {1'b0, 8'hFF, 1'b1, 22'd0};
    wire [31:0] fp_inf_result = {exceptions.sign, 8'hFF, 23'd0};
    wire [31:0] fp_zero_result = {sum_sign, 8'd0, 23'd0};
    wire [31:0] fp_overflow_result = {sum_sign, 8'hFF, 23'd0};
    wire [31:0] fp_normal_result = {sum_sign, packed_exp, final_man};

    logic [31:0] fp_result;
    always_comb begin
        if (exceptions.is_nan) begin
            fp_result = fp_nan_result;
        end else if (exceptions.is_inf) begin
            fp_result = fp_inf_result;
        end else begin
            if (zero_sum || exp_underflow) begin
                 fp_result = fp_zero_result;
            end else if (exp_overflow) begin
                 fp_result = fp_overflow_result;
            end else begin
                 fp_result = fp_normal_result;
            end
        end
    end

`ifdef TCU_TFR_INT_ENABLE
    // Integer handling: reconstruct the two's-complement sum
    wire [WA-1:0] acc_tc = sum_sign ? (-abs_sum) : abs_sum;

    wire [6:0] ext_acc_int = 7'($signed(acc_tc[WA-1:25]));
    wire [6:0] int_hi;
    VX_ks_adder #(
        .N (7),
        .BYPASS (`FORCE_BUILTIN_ADDER(7))
    ) int_adder (
        .cin   (0),
        .dataa (ext_acc_int),
        .datab (cval_hi),
        .sum   (int_hi),
        `UNUSED_PIN (cout)
    );

    wire [31:0] int_result = {int_hi, acc_tc[24:0]};

    assign result = is_int ? int_result : fp_result;
`else
    assign result = fp_result;
`endif

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (valid_in && !is_int && !zero_sum
         && !exceptions.is_nan && !exceptions.is_inf) begin
            `ASSERT(shifted_sum[WA-1],
                ("%t: %s FEDP-NORM(%0d): normalized MSB missing, mag=0x%0h lzc=%0d",
                 $time, INSTANCE_ID, req_id, acc_sig, lz_count));
        end
    end
`endif

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-NORM(%0d): is_int=%b, abs_sum=0x%0h, sign=%b, lzc=%0d, norm_exp=%0d, shifted=0x%0h, R=%b, S=%b, Rup=%b, carry=%b, final_exp=%0d, result=0x%0h\n",
                $time, INSTANCE_ID, req_id, is_int, abs_sum, sum_sign, lz_count, norm_exp_base, shifted_sum, round_bit, sticky_bit, round_up, carry_out, final_exp_s, result));
        end
    end
`endif

endmodule
