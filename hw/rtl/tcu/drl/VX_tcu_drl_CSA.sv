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

module VX_tcu_drl_CSA #(
    parameter M = 4,    //no. of operands
    parameter N = 8     //bit-width of each operand
) (
    input [N-1:0] Operands [M-1:0],
`ifdef CSA_ACC_COUT
    output [N+$clog2(M)-1:0] Sum
`elsif CSA_NO_COUT
    output [N-1:0] Sum
`endif
);
    localparam SUM_WIDTH = N + $clog2(M);
    wire [SUM_WIDTH-1:0] St[M-3:0], Ct[M-3:0]; //2d matrix of size (M-2)x(SUM_WIDTH-1)

    CSA_level #(.N(N)) CSA0(.A(Operands[0]), .B(Operands[1]), .C(Operands[2]), .St(St[0][N:0]), .Ct(Ct[0][N:0]));
    genvar i;
    generate
        for (i = 1; i < M-2; i=i+1) begin : g_csa_tree
            CSA_level #(.N(N+i)) CSA(.A(St[i-1][N-1+i:0]), .B(Ct[i-1][N-1+i:0]), .C({{i{1'b0}},Operands[2+i]}), .St(St[i][N+i:0]), .Ct(Ct[i][N+i:0]));
        end
    endgenerate

`ifdef CSA_ACC_COUT
    KoggeStoneAdder #(.N(SUM_WIDTH)) RCA0(.A(St[M-3]), .B(Ct[M-3]), .Sum(Sum[SUM_WIDTH-1:0]), .Cout(Sum[SUM_WIDTH]));
`elsif CSA_NO_COUT
    KoggeStoneAdder #(.N(N)) KSA(.A(St[M-3][N-1:0]), .B(Ct[M-3][N-1:0]), .Sum(Sum), .Cout());
`endif
endmodule

module CSA_level #(     //3:2 Compressor based reduction tree level
    parameter N = 4
) (
    input [N-1:0] A, B, C,
    output [N:0] St, Ct
);
    assign Ct[0] = 1'b0;
    genvar i;
    generate
        for (i = 0; i < N; i = i  + 1) begin : g_compress_3_2
            FullAdder CSAU(.a(A[i]), .b(B[i]), .cin(C[i]), .sum(St[i]), .cout(Ct[i+1]));
        end
    endgenerate
    assign St[N] = 1'b0;
endmodule

module KoggeStoneAdder #(
    parameter N = 16
) (
    input [N-1:0] A, B,
    output [N-1:0] Sum,
    output Cout
);
    localparam LEVELS = $clog2(N);
    wire [N-1:0] G [LEVELS:0];  //generate signals
    wire [N-1:0] P [LEVELS:0];  //propagate signals
    
    genvar i, k;
    //Level 0: Initial generate and propagate
    generate
        for (i = 0; i < N; i = i + 1) begin: g_initial_gp
            assign G[0][i] = A[i] & B[i];
            assign P[0][i] = A[i] ^ B[i];
        end
    endgenerate
    //Kogge-Stone tree levels
    generate
        for (k = 1; k <= LEVELS; k = k + 1) begin: g_ks_levels
            localparam STEP = 1 << (k-1);   //1, 2, 4, 8 ...
            
            for (i = 0; i < N; i = i + 1) begin: ks_nodes
                if (i >= STEP) begin: compute_gp
                    assign G[k][i] = G[k-1][i] | (P[k-1][i] & G[k-1][i-STEP]);
                    assign P[k][i] = P[k-1][i] & P[k-1][i-STEP];
                end else begin: passthrough_gp
                    assign G[k][i] = G[k-1][i];
                    assign P[k][i] = P[k-1][i];
                end
            end
        end
    endgenerate
    //Final stage sum
    assign Sum[0] = P[0][0];
    generate
        for (i = 1; i < N; i = i + 1) begin: g_final_sum
            assign Sum[i] = P[0][i] ^ G[LEVELS][i-1];
        end
    endgenerate
    assign Cout = G[LEVELS][N-1];    
endmodule

module FullAdder (
    input a, b, cin,
    output sum, cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | ((a ^ b) & cin);
endmodule
