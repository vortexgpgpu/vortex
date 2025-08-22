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

// each 32-bit register packs 4 fp8 operands
// to maintain the same unpacking scheme, and number of operands in the accumulator
// that was used for fp16/bf16, the fp8mul modules also perform a 2 product addition
// so instead of just (a0*b0), these modules perfrom (a0*b0 + a1*b1)

`include "VX_define.vh"

module VX_tcu_drl_bf8mul (
    input  wire enable,
    input  wire [15:0] a,            //two bf8 inputs
    input  wire [15:0] b,
    output logic sign_y,             //fp32 outputs
    output logic [7:0] raw_exp_y,
    output logic [23:0] raw_sig_y    //includes hidden 1 bit
);
    //NOTE: exception handling neglected for now
    `UNUSED_VAR(enable);

    wire sign_a0b0, sign_a1b1;
    wire [5:0] raw_exp_a0b0, raw_exp_a1b1;
    wire [5:0] raw_sig_a0b0, raw_sig_a1b1;

    //a0*b0
    VX_fp8mulE5M2 a0b0mul (
        .a         (a[15:8]),
        .b         (b[15:8]),
        .sign_y    (sign_a0b0),
        .raw_exp_y (raw_exp_a0b0),
        .raw_sig_y (raw_sig_a0b0)
    );

    //a1*b1
    VX_fp8mulE5M2 a1b1mul (
        .a         (a[7:0]),
        .b         (b[7:0]),
        .sign_y    (sign_a1b1),
        .raw_exp_y (raw_exp_a1b1),
        .raw_sig_y (raw_sig_a1b1)
    );

    //a0b0 + a1b1

    //Determine operand with larger exponent and fp32 exp conv
    wire [6:0] raw_exp_diff = {1'b0, raw_exp_a1b1} - {1'b0, raw_exp_a0b0};
    wire exp_a0b0_larger = raw_exp_diff[6];

    wire [5:0] raw_exp_y_fp8 = exp_a0b0_larger ? raw_exp_a0b0 : raw_exp_a1b1;
    wire [7:0] fp8e5m2_conv_bias_fp32 = 8'd99;    //127-30+2
    VX_ks_adder #(
        .N(8)
    ) expconvfp8e5m2 (
        .dataa ({2'd0, raw_exp_y_fp8}),
        .datab (fp8e5m2_conv_bias_fp32),
        .sum   (raw_exp_y),    //fp32 exp out
        `UNUSED_PIN (cout)
    );

    //Align significands by shifting smaller one right
    wire [6:0] shift_amount = exp_a0b0_larger ? -raw_exp_diff : raw_exp_diff;
    wire [22:0] aligned_sig_a0b0 = exp_a0b0_larger ? {raw_sig_a0b0, 17'd0} : ({raw_sig_a0b0, 17'd0} >> shift_amount);
    wire [22:0] aligned_sig_a1b1 = exp_a0b0_larger ? ({raw_sig_a1b1, 17'd0} >> shift_amount) : {raw_sig_a1b1, 17'd0};

    //Converting to signed based on sign bits
    wire [23:0] signed_sig_a0b0 = sign_a0b0 ? -aligned_sig_a0b0 : {1'b0, aligned_sig_a0b0};
    wire [23:0] signed_sig_a1b1 = sign_a1b1 ? -aligned_sig_a1b1 : {1'b0, aligned_sig_a1b1};

    //Signed significand addtion
    wire [24:0] signed_sig_result;
    VX_ks_adder #(
        .N(24)
    ) sig_add (
        .dataa (signed_sig_a0b0),
        .datab (signed_sig_a1b1),
        .sum   (signed_sig_result[23:0]),
        .cout  (signed_sig_result[24])
    );

    //Extracting magnitude from signed result
    wire sum_sign = signed_sig_result[24];
    wire [23:0] abs_sum = sum_sign ? -signed_sig_result[23:0] : signed_sig_result[23:0];

    //Assign raw outputs
    assign sign_y = sum_sign;
    assign raw_sig_y = abs_sum;
endmodule

// just does a * b
module VX_fp8mulE5M2 (
    input wire [7:0] a,
    input wire [7:0] b,
    output logic sign_y,
    output logic [5:0] raw_exp_y,    //true val + 30
    output logic [5:0] raw_sig_y     //deal with 0 extend later
);

    //Extract fields from inputs
    wire sign_a = a[7];
    wire sign_b = b[7];
    wire [4:0] exp_a = a[6:2];
    wire [4:0] exp_b = b[6:2];
    wire [1:0] frac_a = a[1:0];
    wire [1:0] frac_b = b[1:0];

    //Result sign
    assign sign_y = sign_a ^ sign_b;

    //Raw result exponent calculation
    VX_ks_adder #(
        .N(5)
    ) raw_exp_add (
        .dataa (exp_a),
        .datab (exp_b),
        .sum   (raw_exp_y[4:0]),
        .cout  (raw_exp_y[5])
    );

    //Mantissa Calculation
    wire [2:0] full_mant_a = {1'b1, frac_a};
    wire [2:0] full_mant_b = {1'b1, frac_b};
    wire [5:0] product_mant; // = full_mant_a * full_mant_b; //double width signigicand mul
    VX_wallace_mul #(
        .N (3)
    ) wtmul_fp16 (
        .a (full_mant_a),
        .b (full_mant_b),
        .p (product_mant)
    );
    assign raw_sig_y = product_mant;
endmodule