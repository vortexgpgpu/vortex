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

// each 32-bit register packs 4 int8 operands
// to maintain the same unpacking scheme, and number of operands in the accumulator
// that were used for fp16/bf16, the int8mul modules also perform a 2 product addition
// so instead of just (a0*b0), these modules perfrom (a0*b0 + a1*b1)

`include "VX_define.vh"

module VX_tcu_drl_int8mul (
    input  wire enable,
    input  wire [15:0] a,            //two int8 inputs
    input  wire [15:0] b,
    output logic [16:0] signed_y     //int17 result
);
    `UNUSED_VAR(enable);

    wire [1:0][15:0] prod_ab;

    //generate products
    for (genvar i = 0; i < 2; i++) begin : g_prod_ab
        //operand sub-part select
        wire [7:0] a_sel = a[8*i+7 -: 8];
        wire [7:0] b_sel = b[8*i+7 -: 8];

        wire [7:0] a_abs = a_sel[7] ? -a_sel: a_sel;
        wire [7:0] b_abs = b_sel[7] ? -b_sel: b_sel;
        wire ab_sign  = a_sel[7] ^ b_sel[7];
        wire [15:0] prod_abs;

        VX_wallace_mul #(
            .N (8)
        ) mul_ab (
            .a (a_abs),
            .b (b_abs),
            .p (prod_abs)
        ); 

        assign prod_ab[i] = ab_sign ? -prod_abs : prod_abs;
    end

    //a0b0 + a1b1
    assign signed_y = $signed(prod_ab[0]) + $signed(prod_ab[1]);

endmodule
