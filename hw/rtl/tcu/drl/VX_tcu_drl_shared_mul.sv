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

module VX_tcu_drl_shared_mul (
    input wire enable,
    input wire [3:0] fmt_s,
    input wire [15:0] a,
    input wire [15:0] b,
    input wire exp_low_larger,    //from exp_bias module
    input wire [6:0] raw_exp_diff,
    output logic [24:0] y
);
    `UNUSED_VAR(enable);

    //fp16/bf16 pack 2 ops/reg --> need one instantiation per multiplier slice
    wire sign_f16 = a[15] ^ b[15];
    wire [10:0] a_f16 = fmt_s[0] ? {1'b1, a[9:0]} : {3'd0, 1'b1, a[6:0]};
    wire [10:0] b_f16 = fmt_s[0] ? {1'b1, b[9:0]} : {3'd0, 1'b1, b[6:0]};
    wire [21:0] y_f16;
    VX_wallace_mul #(
        .N (11)
    ) wtmul_f16 (
        .a (a_f16),
        .b (b_f16),
        .p (y_f16)
    );

    //fp8/bf8 pack 4 ops/ref --> need two instantiations per multiplier slice
    wire [1:0] sign_f8;
    wire [1:0][7:0] y_f8;
    for (genvar i = 0; i < 2; i++) begin  :  g_f8_mul
        assign sign_f8[i] = a[(i*8)+7] ^ b[(i*8)+7];
        wire [3:0] a_f8 = fmt_s[0] ? {1'b1, a[(i*8)+2 -: 3]} : {1'd0, 1'b1, a[(i*8)+1 -: 2]};
        wire [3:0] b_f8 = fmt_s[0] ? {1'b1, b[(i*8)+2 -: 3]} : {1'd0, 1'b1, b[(i*8)+1 -: 2]};
        VX_wallace_mul #(
            .N (4)
        ) wtmul_f8 (
            .a (a_f8),
            .b (b_f8),
            .p (y_f8[i])
        );
    end
    wire [6:0] shift_amount = exp_low_larger ?  -raw_exp_diff : raw_exp_diff;
    wire [7:0] y_f8_low  = fmt_s[0] ? y_f8[0] : {y_f8[0][5:0], 2'd0};
    wire [7:0] y_f8_high = fmt_s[0] ? y_f8[1] : {y_f8[1][5:0], 2'd0};
    wire [22:0] aligned_sig_low  = exp_low_larger ? {y_f8_low, 15'd0} : {y_f8_low, 15'd0} >> shift_amount;
    wire [22:0] aligned_sig_high = exp_low_larger ? {y_f8_high, 15'd0} >> shift_amount : {y_f8_high, 15'd0};
    wire [23:0] signed_sig_low  = sign_f8[0] ? -aligned_sig_low  : {1'b0, aligned_sig_low};
    wire [23:0] signed_sig_high = sign_f8[1] ? -aligned_sig_high : {1'b0, aligned_sig_high};
    wire [24:0] signed_sig_res;
    VX_ks_adder #(
        .N(24)
    ) sig_adder_f8 (
        .dataa (signed_sig_low),
        .datab (signed_sig_high),
        .sum   (signed_sig_res[23:0]),
        .cout  (signed_sig_res[24])
    );
    wire sign_f8_add = signed_sig_res[24];
    wire [23:0] y_f8_add = sign_f8_add ? -signed_sig_res[23:0] : signed_sig_res[23:0];

    //int8 pack 4 ops/reg --> need two instantiations per multiplier slice
    wire [1:0][16:0] y_i8;
    for (genvar i = 0; i < 2; i++) begin : g_i8_mul
        wire [7:0] a_i8 = a[8*i+7 -: 8];
        wire [7:0] b_i8 = b[8*i+7 -: 8];
        wire [7:0] abs_a_i8 = fmt_s[0] ? (a_i8[7] ? -a_i8 : a_i8) : a_i8;
        wire [7:0] abs_b_i8 = fmt_s[0] ? (b_i8[7] ? -b_i8 : b_i8) : b_i8;
        wire ab_sign_i8 = a_i8[7] ^ b_i8[7];
        wire [15:0] abs_y_i8;
        VX_wallace_mul #(
            .N (8)
        ) wtmul_i8 (
            .a (abs_a_i8),
            .b (abs_b_i8),
            .p (abs_y_i8)
        );
        wire [15:0] signed_y_i8 = ab_sign_i8 ? -abs_y_i8 : abs_y_i8;
        assign y_i8[i] = fmt_s[0] ? 17'($signed(signed_y_i8)) : {1'b0, abs_y_i8};
    end
    wire [16:0] y_i8_add;
    //KSA needs to be 17-bit for carry overflow handling 
    VX_ks_adder #(
        .N(17)
    ) i8_adder (
        .dataa (y_i8[0]),
        .datab (y_i8[1]),
        .sum   (y_i8_add),
        `UNUSED_PIN(cout)
    );

    //int4 pack 8 ops/reg --> need four instantiations per multiplier slice
    wire [3:0][9:0] y_i4;
    for (genvar i = 0; i < 4; i++) begin : g_i4_mul
        wire [3:0] a_i4 = a[4*i+3 -: 4];
        wire [3:0] b_i4 = b[4*i+3 -: 4];
        wire [3:0] abs_a_i4 = fmt_s[0] ? (a_i4[3] ? -a_i4 : a_i4) : a_i4;
        wire [3:0] abs_b_i4 = fmt_s[0] ? (b_i4[3] ? -b_i4 : b_i4) : b_i4;
        wire ab_sign_i4 = a_i4[3] ^ b_i4[3];
        wire [7:0] abs_y_i4;
        VX_wallace_mul #(
            .N (4)
        ) wtmul_i4 (
            .a (abs_a_i4),
            .b (abs_b_i4),
            .p (abs_y_i4)
        );
        wire [7:0] signed_y_i4 = ab_sign_i4 ? -abs_y_i4 : abs_y_i4;
        assign y_i4[i] = fmt_s[0] ? 10'($signed(signed_y_i4)) : {2'd0, abs_y_i4};
    end
    wire [9:0] y_i4_add;
    VX_csa_tree #(
        .N (4),
        .W (10),    //8+log2(4)
        .S (10)
    ) i4_adder (
        .operands (y_i4),
        .sum      (y_i4_add),
        `UNUSED_PIN (cout)
    );

    //Select sign+sig out based on datatype
    always_comb begin
        case (fmt_s)
            4'd1:  y = {sign_f16, y_f16, 2'd0};          //fp16
            4'd2:  y = {sign_f16, y_f16[15:0], 8'd0};    //bf16
            4'd3:  y = {sign_f8_add, y_f8_add};          //fp8
            4'd4:  y = {sign_f8_add, y_f8_add};          //bf8
            4'd9:  y = 25'($signed(y_i8_add));           //int8
            4'd10: y = {8'd0, y_i8_add};                 //uint8
            4'd11: y = 25'($signed(y_i4_add));           //int4
            4'd12: y = {15'd0, y_i4_add};                //uint4
            default: y = 25'hxxxxxxx;
        endcase
    end

endmodule
