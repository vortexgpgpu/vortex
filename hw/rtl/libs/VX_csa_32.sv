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

`TRACING_OFF

// Full Adder module
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

// 3:2 Compressor level
module VX_csa_32 #(
    parameter N = 3,
    parameter WIDTH_O = N + 2
) (
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    input  wire [N-1:0] c,
    output wire [WIDTH_O-1:0] sum,
    output wire [WIDTH_O-1:0] carry
);
    wire [N-1:0] sum_int;
    wire [N-1:0] carry_int;

    for (genvar i = 0; i < N; i++) begin : g_compress_3_2
        FullAdder FA (
            .a    (a[i]),
            .b    (b[i]),
            .cin  (c[i]),
            .sum  (sum_int[i]),
            .cout (carry_int[i])
        );
    end

    assign sum = WIDTH_O'(sum_int);
    assign carry = WIDTH_O'({1'b0, carry_int, 1'b0});
endmodule

`TRACING_ON
