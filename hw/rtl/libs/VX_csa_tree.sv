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
    input  wire [N-1:0] c,
    output wire [N:0]   sum,
    output wire [N:0]   carry
);
    for (genvar i = 0; i < N; i++) begin : g_compress_3_2
        FullAdder FA (
            .a    (a[i]),
            .b    (b[i]),
            .cin  (c[i]),
            .sum  (sum[i]),
            .cout (carry[i+1])
        );
    end

    assign carry[0] = 1'b0;
    assign sum[N] = 1'b0;

endmodule

module VX_csa_tree #(
    parameter N = 4,  // Number of operands
    parameter W = 8,  // Bit-width of each operand
    parameter S = W + $clog2(N)  // Output width
) (
    input  wire [N-1:0][W-1:0] operands,  // Input operands
    output wire [S-1:0] sum,  // Final sum output
    output wire cout
);
    `STATIC_ASSERT (N >= 3, ("N must be at least 3"));
    localparam LEVELS = N - 2; // Number of levels in the CSA tree
    localparam WN = W + LEVELS;

    wire [WN-1:0] St [0:LEVELS];
    wire [WN-1:0] Ct [0:LEVELS];

    assign St[0] = WN'(operands[0]);
    assign Ct[0] = WN'(operands[1]);

    for (genvar i = 0; i < LEVELS; i++) begin : g_csa_tree
        localparam WI = W + i;
        wire [WI:0] st, ct;
        CSA_level #(
            .N (WI)
        ) CSA (
            .a    (WI'(St[i])),
            .b    (WI'(Ct[i])),
            .c    (WI'(operands[2+i])),
            .sum  (st),
            .carry(ct)
        );
        assign St[i+1] = WN'(st);
        assign Ct[i+1] = WN'(ct);
    end

    VX_ks_adder #(
        .N (S)
    ) KSA (
        .dataa (St[LEVELS][S-1:0]),
        .datab (Ct[LEVELS][S-1:0]),
        .sum   (sum),
        .cout  (cout)
    );

endmodule

`TRACING_ON
