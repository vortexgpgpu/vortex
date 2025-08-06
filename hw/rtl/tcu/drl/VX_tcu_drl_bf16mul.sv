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

module VX_tcu_drl_bf16mul (
    input  wire enable,
    input  wire [15:0] a, // bf16 input
    input  wire [15:0] b,
    output logic [31:0] y // fp32 output
);
    // NOTE: exception handling neglected for now
    `UNUSED_VAR(enable);

    //Extract fields from inputs
    wire sign_a = a[15];
    wire sign_b = b[15];
    wire [7:0] exp_a = a[14:7];
    wire [7:0] exp_b = b[14:7];
    wire [6:0] frac_a = a[6:0];
    wire [6:0] frac_b = b[6:0];

    //Result sign
    wire result_sign = sign_a ^ sign_b;

    //Result Mantissa Calculation
    wire [7:0] full_mant_a = {1'b1, frac_a};
    wire [7:0] full_mant_b = {1'b1, frac_b};
    wire [15:0] product_mant; // = full_mant_a * full_mant_b; //double width signigicand mul

    VX_wallace_mul #(
        .N (8)
    ) wtmul_bf16 (
        .a (full_mant_a),
        .b (full_mant_b),
        .p (product_mant)
    );

    //Partial norm for FP32 conversion
    wire normalize_shift = product_mant[15];
    wire [22:0] fp32_mantissa = normalize_shift ? {product_mant[14:0], 8'b00000000} : {product_mant[13:0], 9'b000000000};

    //Result Exponent Calculation
    wire [7:0] biased_exp; // = exp_a + exp_b + {7'd0, normalize_shift} - 8'd127;
    wire [7:0] neg_bias = -8'd127;

    VX_csa_tree #(
        .N(4),
        .W(8),
        .S(8)
    ) biasexp_bf16(
        .operands({exp_a, exp_b, {7'd0, normalize_shift}, neg_bias}),
        .sum     (biased_exp)
    );

    `UNUSED_VAR(biased_exp);

    assign y = {result_sign, biased_exp[7:0], fp32_mantissa};
endmodule

//Exception Handling logic
/*
    //Hidden bits (implied 1)
    wire hidden_A = |exp_A; // 0 if exp_A is all zeros (denormal), 1 otherwise
    wire hidden_B = |exp_B;

    //Exception/Special Cases
    wire A_is_zero = (~hidden_A) & (~|frac_A);
    wire B_is_zero = (~hidden_B) & (~|frac_B);
    wire A_is_inf = (&exp_A) & (~|frac_A);
    wire B_is_inf = (&exp_B) & (~|frac_B);
    wire A_is_nan = (&exp_A) & (|frac_A);
    wire B_is_nan = (&exp_B) & (|frac_B);

    wire underflow = exp_sum[9] | (~|exp_sum[8:0]); // Underflow detected --> Negative result or 0
    wire overflow = ~(exp_sum[9]) & (exp_sum[8] | (&exp_sum[7:0])); // Overflow detected if +ve exp and exp >= 255

    //Result selection flags
    wire result_is_nan = A_is_nan | B_is_nan | (A_is_inf & B_is_zero) | (B_is_inf & A_is_zero);
    wire result_is_inf = overflow | (A_is_inf & ~B_is_zero) | (B_is_inf & ~A_is_zero);
    wire result_is_zero = underflow | A_is_zero | B_is_zero;

    //Final result logic
    logic [7:0] final_exp;
    logic [22:0] final_frac;
    always_comb begin
        case({result_is_nan, result_is_inf, result_is_zero})
            3'b100: begin
                        final_exp = 8'hFF;
                        final_frac = 23'h400000;  //Canonical NaN
            end
            3'b010: begin
                        final_exp = 8'hFF;
                        final_frac = 23'h000000;  //Infinity
            end
            3'b001: begin
                        final_exp = 8'h00;
                        final_frac = 23'h000000;  //Zero
            end
            default: begin
                        final_exp = exp_sum[7:0];
                        final_frac = fp32_mantissa;
            end
        endcase
    end

    assign y = {result_sign, final_exp, final_frac};
*/
