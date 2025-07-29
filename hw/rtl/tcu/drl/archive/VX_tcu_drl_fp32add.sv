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

module VX_tcu_drl_fp32add (
    input wire enable,
    input wire [31:0] a, //fp32 input
    input wire [31:0] b,
    output logic [31:0] y //fp32 output
);

    // NOTE: no exception handling atm
    `UNUSED_VAR(enable);

    //Extracting fields from inputs
    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];
    wire [7:0] exp_b = b[30:23];
    wire [22:0] frac_a = a[22:0];
    wire [22:0] frac_b = b[22:0];
    
    //Full mantissas with implicit 1
    wire [23:0] full_sig_a = {1'b1, frac_a};
    wire [23:0] full_sig_b = {1'b1, frac_b};
    
    //Determining which operand has larger exponent
    wire exp_a_larger = (exp_a >= exp_b);
    wire [7:0] exp_diff = exp_a_larger ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [7:0] exp_result = exp_a_larger ? exp_a : exp_b;
    
    //Aligning sigissas by shifting smaller one right
    wire [23:0] aligned_sig_a = exp_a_larger ? full_sig_a : (full_sig_a >> exp_diff);
    wire [23:0] aligned_sig_b = exp_a_larger ? (full_sig_b >> exp_diff) : full_sig_b;
    
    //Converting to signed based on sign bits
    wire signed [24:0] signed_sig_a = sign_a ? -aligned_sig_a : {1'b0, aligned_sig_a};
    wire signed [24:0] signed_sig_b = sign_b ? -aligned_sig_b : {1'b0, aligned_sig_b};
    
    //Signed addition
    wire signed [25:0] signed_sum_sig = signed_sig_a + signed_sig_b;
    
    //Sign and magnitude extraction
    wire result_sign = signed_sum_sig[25];
    logic [24:0] sum_sig;
    logic [7:0] final_exp;
    
    always_comb begin
        sum_sig = result_sign ? -signed_sum_sig[24:0] : signed_sum_sig[24:0];
        final_exp = exp_result;
        
        // Normalization (Overflow: shift right and increment exponent)
        if (sum_sig[24]) begin
            final_exp = exp_result + 1'b1;
            sum_sig = sum_sig >> 1;
        end
    end
    
    assign y = {result_sign, final_exp, sum_sig[22:0]};
endmodule
