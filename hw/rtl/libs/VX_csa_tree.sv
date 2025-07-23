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
`IGNORE_UNOPTFLAT_BEGIN
    output wire sum,
    output wire cout
`IGNORE_UNOPTFLAT_END
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
    input  wire [W-1:0] operands [N-1:0],  // Input operands
    output wire [S-1:0] sum  // Final sum output
);
    `STATIC_ASSERT (N >= 3, ("N must be at least 3"));
    localparam PP_ENABLE = (S == W); // Partial product flag
    localparam LEVELS = $clog2(N-2); // Number of levels in the CSA tree
    localparam SUM_WIDTH = W + $clog2(N);

    wire [SUM_WIDTH-1:0] St[N-3:0], Ct[N-3:0]; //2d matrix of size (N-2)x(SUM_WIDTH-1)

    CSA_level #(
        .N (W)
    ) CSA0 (
        .a    (operands[0]),
        .b    (operands[1]),
        .c    (operands[2]),
        .sum  (St[0][W:0]),
        .carry(Ct[0][W:0])
    );

    for (genvar i = 1; i < LEVELS; i++) begin : g_csa_tree
        CSA_level #(
            .N (W+i)
        ) CSA (
            .a    (St[i-1][W-1+i:0]),
            .b    (Ct[i-1][W-1+i:0]),
            .c    ({{i{1'b0}}, operands[2+i]}),
            .sum  (St[i][W+i:0]),
            .carry(Ct[i][W+i:0])
        );
    end

    if (PP_ENABLE) begin : g_pp_adder
        VX_ks_adder #(
            .N (S)
        ) KSA (
            .dataa (St[N-3][W-1:0]),
            .datab (Ct[N-3][W-1:0]),
            .sum   (sum),
            `UNUSED_PIN (cout)
        );
    end else begin : g_rc_adder
        VX_ks_adder #(
            .N (S)
        ) RCA0 (
            .dataa (St[N-3]),
            .datab (Ct[N-3]),
            .sum   (sum),
            `UNUSED_PIN (cout)
        );
    end

endmodule
