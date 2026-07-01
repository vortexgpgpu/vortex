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

module VX_tcu_tet_norm_round import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WA = 30,
    parameter EXP_W = 10,
    parameter C_HI_W = 8
) (
    input wire          clk,
    input wire          reset,
    input wire          enable,
    input wire          valid_in,
    input wire [31:0]   req_id,
    input wire [EXP_W-1:0] max_exp,
    input wire [WA-1:0] acc_sig,
    input wire [C_HI_W-1:0] cval_hi,
    input wire          is_int,
    input wire          sticky_in,
    input fedp_excep_t  exceptions,
    output wire [31:0]  result
);
    `UNUSED_SPARAM (INSTANCE_ID)

`ifdef VX_CFG_TCU_INT8_ENABLE
`define TET_NORM_INT_ENABLE
`elsif VX_CFG_TCU_INT4_ENABLE
`define TET_NORM_INT_ENABLE
`endif

    wire sum_sign = acc_sig[WA-1];
    wire [WA-1:0] abs_sum = sum_sign ? (~acc_sig + WA'(1)) : acc_sig;
    wire zero_sum = ~|abs_sum;

    wire [$clog2(WA)-1:0] lz_count_pred;
    VX_lzc #(
        .N(WA)
    ) lzc_inst (
        .data_in   (abs_sum),
        .data_out  (lz_count_pred),
        `UNUSED_PIN (valid_out)
    );

    wire signed [EXP_W-1:0] norm_exp_base;
    wire signed [EXP_W-1:0] norm_exp_plus1;
    wire signed [EXP_W-1:0] norm_exp_plus2;

    wire [EXP_W-1:0] sub_term = EXP_W'({1'b1, 2'b00, lz_count_pred});
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
    assign norm_exp_plus2 = norm_exp_base + EXP_W'(2'd2);

    wire [WA:0] abs_sum_ext = {1'b0, abs_sum};
    wire [WA:0] shifted_sum_raw = abs_sum_ext << lz_count_pred;

    wire overshift = shifted_sum_raw[WA];

    wire [26:0] aligned_bits = overshift ? shifted_sum_raw[WA   -: 27]
                                         : shifted_sum_raw[WA-1 -: 27];

    wire [23:0] norm_man   = aligned_bits[26:3];
    wire        guard_bit  = aligned_bits[2];
    wire        round_bit  = aligned_bits[1];
    wire        sticky_bit = aligned_bits[0] | (|shifted_sum_raw[WA-27:0]) | sticky_in;
    wire        lsb_bit    = norm_man[0];
    wire        round_up   = guard_bit && (round_bit || sticky_bit || lsb_bit);

`ifdef TET_NORM_INT_ENABLE
    wire [6:0] ext_acc_int = 7'($signed(acc_sig[WA-1:25]));
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

    wire [31:0] int_result = {int_hi, acc_sig[24:0]};
`else
    `UNUSED_VAR ({cval_hi, is_int})
`endif

    wire [23:0] s1_norm_man;
    wire        s1_round_up;
    wire        s1_overshift;
    wire signed [EXP_W-1:0] s1_norm_exp_base;
    wire signed [EXP_W-1:0] s1_norm_exp_plus1;
    wire signed [EXP_W-1:0] s1_norm_exp_plus2;
    wire        s1_sum_sign;
    wire        s1_zero_sum;
    fedp_excep_t s1_exceptions;
    wire        s1_is_int;
`ifdef TET_NORM_INT_ENABLE
    wire [31:0] s1_int_result;
`endif
    wire [WA-1:0] s1_abs_sum;
    wire [$clog2(WA)-1:0] s1_lz_count_pred;
    wire [WA:0] s1_shifted_sum_raw;
    wire        s1_round_bit;
    wire        s1_sticky_bit;
    wire        s1_valid;
    wire [31:0] s1_req_id;

    VX_tcu_tet_register #(
    `ifdef TET_NORM_INT_ENABLE
        .DATAW (24 + 1 + 1 + (3 * EXP_W) + 1 + 1 + $bits(fedp_excep_t) + 1 + 32 + WA + $clog2(WA) + (WA+1) + 1 + 1 + 1 + 32),
    `else
        .DATAW (24 + 1 + 1 + (3 * EXP_W) + 1 + 1 + $bits(fedp_excep_t) + 1 + WA + $clog2(WA) + (WA+1) + 1 + 1 + 1 + 32),
    `endif
        .DEPTH (1)
    ) pipe_norm0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
    `ifdef TET_NORM_INT_ENABLE
        .data_in  ({norm_man,    round_up,    overshift,    norm_exp_base,    norm_exp_plus1,    norm_exp_plus2,    sum_sign,    zero_sum,    exceptions,    is_int,    int_result,    abs_sum,    lz_count_pred,    shifted_sum_raw,    round_bit,    sticky_bit,    valid_in,  req_id}),
        .data_out ({s1_norm_man, s1_round_up, s1_overshift, s1_norm_exp_base, s1_norm_exp_plus1, s1_norm_exp_plus2, s1_sum_sign, s1_zero_sum, s1_exceptions, s1_is_int, s1_int_result, s1_abs_sum, s1_lz_count_pred, s1_shifted_sum_raw, s1_round_bit, s1_sticky_bit, s1_valid, s1_req_id})
    `else
        .data_in  ({norm_man,    round_up,    overshift,    norm_exp_base,    norm_exp_plus1,    norm_exp_plus2,    sum_sign,    zero_sum,    exceptions,    is_int,    abs_sum,    lz_count_pred,    shifted_sum_raw,    round_bit,    sticky_bit,    valid_in,  req_id}),
        .data_out ({s1_norm_man, s1_round_up, s1_overshift, s1_norm_exp_base, s1_norm_exp_plus1, s1_norm_exp_plus2, s1_sum_sign, s1_zero_sum, s1_exceptions, s1_is_int, s1_abs_sum, s1_lz_count_pred, s1_shifted_sum_raw, s1_round_bit, s1_sticky_bit, s1_valid, s1_req_id})
    `endif
    );

`ifndef DBG_TRACE_TCU
    `UNUSED_VAR ({s1_req_id, s1_valid, s1_round_bit, s1_sticky_bit, s1_abs_sum, s1_lz_count_pred, s1_shifted_sum_raw})
    `ifndef TET_NORM_INT_ENABLE
        `UNUSED_VAR (s1_is_int)
    `endif
`endif

    wire [24:0] man_plus_zero = {1'b0, s1_norm_man};
    wire [24:0] rounded_sig_full;
    VX_ks_adder #(
        .N(25),
        .BYPASS (`FORCE_BUILTIN_ADDER(25))
    ) round_adder (
        .cin   (s1_round_up),
        .dataa (man_plus_zero),
        .datab (25'd0),
        .sum   (rounded_sig_full),
        `UNUSED_PIN (cout)
    );

    wire carry_out = rounded_sig_full[24];
    wire [22:0] final_man = carry_out ? rounded_sig_full[23:1] : rounded_sig_full[22:0];

    logic signed [EXP_W-1:0] final_exp_s;
    always_comb begin
        case ({s1_overshift, carry_out})
            2'b00: final_exp_s = s1_norm_exp_base;
            2'b01: final_exp_s = s1_norm_exp_plus1;
            2'b10: final_exp_s = s1_norm_exp_plus1;
            2'b11: final_exp_s = s1_norm_exp_plus2;
        endcase
    end

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
    wire [31:0] fp_inf_result = {s1_exceptions.sign, 8'hFF, 23'd0};
    wire [31:0] fp_zero_result = {s1_sum_sign, 8'd0, 23'd0};
    wire [31:0] fp_overflow_result = {s1_sum_sign, 8'hFF, 23'd0};
    wire [31:0] fp_normal_result = {s1_sum_sign, packed_exp, final_man};

    logic [31:0] fp_result;
    always_comb begin
        if (s1_exceptions.is_nan) begin
            fp_result = fp_nan_result;
        end else if (s1_exceptions.is_inf) begin
            fp_result = fp_inf_result;
        end else begin
            if (s1_zero_sum || exp_underflow) begin
                 fp_result = fp_zero_result;
            end else if (exp_overflow) begin
                 fp_result = fp_overflow_result;
            end else begin
                 fp_result = fp_normal_result;
            end
        end
    end

`ifdef TET_NORM_INT_ENABLE
    assign result = s1_is_int ? s1_int_result : fp_result;
`else
    assign result = fp_result;
`endif

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (s1_valid) begin
            `TRACE(4, ("%t: %s FEDP-NORM(%0d): is_int=%b, abs_sum=0x%0h, sign=%b, lzc=%0d, norm_exp=%0d, shifted=0x%0h, R=%b, S=%b, Rup=%b, carry=%b, final_exp=%0d, result=0x%0h\n",
                $time, INSTANCE_ID, s1_req_id, s1_is_int, s1_abs_sum, s1_sum_sign, s1_lz_count_pred, s1_norm_exp_base, s1_shifted_sum_raw, s1_round_bit, s1_sticky_bit, s1_round_up, carry_out, final_exp_s, result));
        end
    end
`endif

endmodule

`ifdef TET_NORM_INT_ENABLE
`undef TET_NORM_INT_ENABLE
`endif
