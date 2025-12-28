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
    
module VX_tcu_drl_norm_round #(
    parameter N = 5,     //includes c_val
    parameter W = 53    //acc out width    
) (
    input wire [7:0] max_exp,
    input wire [W+$clog2(N):0] acc_sig,
    input wire [6:0] hi_c,
    input wire [N-2:0] sigSigns,
    input wire fmt_sel,
    input wire [2:0] exceptions,
    output wire [31:0] result
);

    localparam ACC_WIDTH = W+$clog2(N)-1;

    //Extracting magnitude from signed acc sig
    wire sum_sign = acc_sig[W+$clog2(N)-1];
    wire [ACC_WIDTH-1:0] abs_sum;
    assign abs_sum = sum_sign ? -acc_sig[ACC_WIDTH-1:0] : acc_sig[ACC_WIDTH-1:0];
    
    //Exception handling
    wire zero_sum = ~|abs_sum;

    //Leading zero counter
    wire [$clog2(ACC_WIDTH)-1:0] lz_count;
    VX_lzc #(
        .N (ACC_WIDTH)
    ) lzc (
        .data_in   (abs_sum),
        .data_out  (lz_count),
        `UNUSED_PIN(valid_out)
    );

    //Exponent normalization
    wire [7:0] shift_amount = 8'($clog2(N)) - 8'(lz_count);
    wire [7:0] norm_exp = max_exp + shift_amount;

    //Move leading 1 to MSB (mantissa norm)
    wire [ACC_WIDTH-1:0] shifted_acc_sig = abs_sum << lz_count;
    //RNE rounding
    wire lsb = shifted_acc_sig[ACC_WIDTH-24];
    wire guard_bit = shifted_acc_sig[ACC_WIDTH-25];
    wire round_bit = shifted_acc_sig[ACC_WIDTH-26];
    wire sticky_bit = |shifted_acc_sig[ACC_WIDTH-27:0];
    wire round_up = guard_bit & (round_bit | sticky_bit | lsb);    
    //Index [ACC_WIDTH-1] becomes the hidden 1
    wire [22:0] rounded_sig;
    wire carry_out;
    VX_ks_adder #(
        .N (23)
    ) round_up_adder (
        .dataa ({shifted_acc_sig[ACC_WIDTH-2 : ACC_WIDTH-24]}),
        .datab (23'(round_up)),
        .sum   (rounded_sig),
        .cout  (carry_out)
    );

    wire [7:0] adjusted_exp = zero_sum ? 8'd0 : (norm_exp + 8'(carry_out));

    logic [31:0] fp_result;
    always_comb begin
        case (exceptions[1:0])
            2'b00: fp_result = {sum_sign, adjusted_exp, rounded_sig};
            2'b01: fp_result = {exceptions[2], 8'hFF, 23'h000000};
            2'b10: fp_result = {exceptions[2], 8'hFF, 23'h400000};
            default: fp_result = 32'hxxxxxxxx;
        endcase            
    end

    //Final INT addition
    wire [6:0] ext_acc_int = 7'($signed(acc_sig[W+$clog2(N):W]));
    wire [N-2:0][6:0] ext_signs;
    for (genvar i = 0; i < N-1; i++) begin : g_sign_ext
        assign ext_signs[i] = 7'($signed(sigSigns[i]));
    end

    wire [6:0] int_hi;

    VX_csa_tree #(
        .N (N+1),
        .W (7),
        .S (7)
    ) int_adder (
        .operands ({ext_acc_int, hi_c, ext_signs}),
        .sum (int_hi),
        `UNUSED_PIN (cout)
    );

    wire [31:0] int_result = {int_hi, acc_sig[(W-25)+:25]};

    assign result = fmt_sel ? int_result : fp_result;

endmodule
