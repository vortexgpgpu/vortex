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
    input  wire [15:0] a,            //bf16 inputs
    input  wire [15:0] b,
    output logic sign_y,             //fp32 outputs
    output logic [7:0] raw_exp_y,
    output logic [23:0] raw_sig_y    //includes hidden 1 bit
);
    //NOTE: exception handling neglected for now
    `UNUSED_VAR(enable);

    //Extract fields from inputs
    wire sign_a = a[15];
    wire sign_b = b[15];
    wire [7:0] exp_a = a[14:7];
    wire [7:0] exp_b = b[14:7];
    wire [6:0] frac_a = a[6:0];
    wire [6:0] frac_b = b[6:0];

    //Result sign
    assign sign_y = sign_a ^ sign_b;

    //Raw result exponent calculation
    wire [9:0] neg_bias = 10'b1110000010; //-127+1
    wire [9:0] raw_exp_y_signed;
    `UNUSED_VAR (raw_exp_y_signed);
    VX_csa_tree #(
        .N(3),
        .W(10),
        .S(10)
    ) biasexp_bf16(
        .operands({{2'd0, exp_a}, {2'd0, exp_b}, neg_bias}),
        .sum     (raw_exp_y_signed),
        `UNUSED_PIN (cout)
    );
    assign raw_exp_y = raw_exp_y_signed[9] ? -raw_exp_y_signed[7:0] : raw_exp_y_signed[7:0];

    //Mantissa Calculation
    wire [7:0] full_mant_a = {1'b1, frac_a};
    wire [7:0] full_mant_b = {1'b1, frac_b};
    wire [15:0] product_mant;    // = full_mant_a * full_mant_b; //double width signigicand mul
    VX_wallace_mul #(
        .N (8)
    ) wtmul_bf16 (
        .a (full_mant_a),
        .b (full_mant_b),
        .p (product_mant)
    );

    //Zero-extend raw significand (including hidden) to fp32 sig width
    assign raw_sig_y = {product_mant, 8'h00};
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

    //Result Norm req check & Significand
    assign norm_shift = product_mant[15];
    assign sig_y = norm_shift ? {product_mant[14:0], 8'b00000000} : {product_mant[13:0], 9'b000000000};

    assign y = {result_sign, final_exp, final_frac};
*/
