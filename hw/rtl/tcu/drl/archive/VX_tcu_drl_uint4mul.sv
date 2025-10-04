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

// each 32-bit register packs 8 uint4 operands
// to maintain the same unpacking scheme, and number of operands in the accumulator
// that were used for fp16/bf16, the uint4mul modules also perform a 4 product addition
// so instead of just (a0*b0), these modules perfrom (a0*b0 + a1*b1 + a2*b2 + a3*b3)

`include "VX_define.vh"

module VX_tcu_drl_uint4mul (
    input  wire enable,
    input  wire [15:0] a,            //two int8 inputs
    input  wire [15:0] b,
    output logic [9:0] unsigned_y     //8 + $clog2(4)
);
    `UNUSED_VAR(enable);

    wire [3:0][7:0] prod_ab;

    //generate products
    for (genvar i = 0; i < 4; i++) begin : g_prod_ab
           VX_wallace_mul #(
            .N (4)
        ) mul_ab (
            .a (a[4*i+3 -: 4]),
            .b (b[4*i+3 -: 4]),
            .p (prod_ab[i])
        ); 
    end

    //accumulate products
    VX_csa_tree #(
        .N (4),
        .W (8),
        .S (10)
    ) prod_acc (
        .operands (prod_ab),
        .sum      (unsigned_y),
        `UNUSED_PIN (cout)
    );

endmodule
