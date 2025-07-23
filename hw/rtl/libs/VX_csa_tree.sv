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

module FullAdder (
    input  wire a,
    input  wire b,
    input  wire cin,
    output wire sum,
    output wire cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | ((a ^ b) & cin);

endmodule

// 3:2 Compressor based reduction tree level
module CSA_level #(
    parameter N = 4
) (
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    input  wire [N-1:0] cin,
    output wire [N:0]   sum,
    output wire [N:0]   cout
);
    for (genvar i = 0; i < N; i++) begin : g_compress_3_2
        FullAdder FA (
            .a    (a[i]),
            .b    (b[i]),
            .cin  (cin[i]),
            .sum  (sum[i]),
            .cout (cout[i+1])
        );
    end
    assign cout[0] = 1'b0;
    assign sum[N]  = 1'b0;

endmodule

module VX_csa_tree #(
    parameter N   = 4,  // no. of operands
    parameter W   = 8,  // bit-width of each operand
    parameter CEN = 1,  // carry enable
    parameter SW  = W + $clog2(N),
    parameter SCW = CEN ? SW : W
) (
    input [W-1:0]    operands [N-1:0],
    output [SCW-1:0] sum
);
    wire [SW-1:0] St [N-3:0];
    wire [SW-1:0] Ct [N-3:0];

    CSA_level #(
        .N (W)
    ) CSA0 (
        .a    (operands[0]),
        .b    (operands[1]),
        .cin  (operands[2]),
        .sum  (St[0][W:0]),
        .cout (Ct[0][W:0])
    );

    for (genvar i = 1; i < N-2; i++) begin : g_csa_tree
        CSA_level #(
            .N (W+i)
        ) CSA (
            .a    (St[i-1][W-1+i:0]),
            .b    (Ct[i-1][W-1+i:0]),
            .cin  ({{i{1'b0}}, operands[2+i]}),
            .sum  (St[i][W+i:0]),
            .cout (Ct[i][W+i:0])
        );
    end

    if (CEN != 0) begin : g_cout
        VX_ks_adder #(
            .N (SCW)
        ) RCA0 (
            .dataa (St[N-3]),
            .datab (Ct[N-3]),
            .sum   (sum[SW-2:0]),
            .cout  (sum[SW-1])
        );
    end else begin : g_no_cout
        VX_ks_adder #(
            .N (SCW)
        ) RCA0 (
            .dataa (St[N-3][W-1:0]),
            .datab (Ct[N-3][W-1:0]),
            .sum   (sum),
            `UNUSED_PIN(cout)
        );
    end

endmodule
