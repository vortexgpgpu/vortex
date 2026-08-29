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

// 5:3 counter for 4:2 compressor
module counter_5to3(
    input wire x1, x2, x3, x4, cin,
    output wire sum, carry, cout
);
    // FA1: x1 + x2 + x3
    wire s1 = x1 ^ x2 ^ x3;
    assign cout = (x1 & x2) | (x2 & x3) | (x1 & x3);

    // FA2: s1 + x4 + cin
    assign sum = s1 ^ x4 ^ cin;
    assign carry = (s1 & x4) | (x4 & cin) | (s1 & cin);
endmodule

// 4:2 Compressor level
module VX_csa_42 #(
    parameter N = 4,
    parameter WIDTH_O = N + 2
) (
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    input  wire [N-1:0] c,
    input  wire [N-1:0] d,
    output wire [WIDTH_O-1:0] sum,
    output wire [WIDTH_O-1:0] carry
);
    wire [N-1:0] sum_int;
    wire [N:0]   cin;
    wire [N-1:0] cout;
    wire [N-1:0] carry_int;

    assign cin[0] = 1'b0;

    // Cascaded 5:3 counters
    for (genvar i = 0; i < N; i++) begin : g_compress_4_2
        counter_5to3 u_counter_5to3(
            .x1(a[i]),
            .x2(b[i]),
            .x3(c[i]),
            .x4(d[i]),
            .cin(cin[i]),
            .sum(sum_int[i]),
            .carry(carry_int[i]),
            .cout(cout[i])
        );
        assign cin[i+1] = cout[i];
    end

    wire [1:0] carry_temp;

    assign sum = WIDTH_O'(sum_int);
    assign carry_temp = {carry_int[N-1] & cin[N], carry_int[N-1] ^ cin[N]};
    assign carry = WIDTH_O'({carry_temp, carry_int[N-2:0], 1'b0});

endmodule

`TRACING_ON

