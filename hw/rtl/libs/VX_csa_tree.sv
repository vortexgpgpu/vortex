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

// 5:3 counter for 4:2 compressor
module counter_5to3(
    input logic x1, x2, x3, x4, cin,
    output logic sum, carry, cout
);
    assign sum = x1 ^ x2 ^ x3 ^ x4 ^ cin;
    assign cout = (x1 ^ x2) & x3 | ~(x1 ^ x2) & x1;
    assign carry = (x1 ^ x2 ^ x3 ^ x4) & cin | ~(x1 ^ x2 ^ x3 ^ x4) & x4;
endmodule

// 3:2 Compressor level
module CSA_level_3to2 #(
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

// 4:2 Compressor level
module CSA_level_4to2 #(
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
    wire [N:0] cin;
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
    assign carry_temp = carry_int[N-1] + cin[N];
    assign carry = WIDTH_O'({carry_temp, carry_int[N-2:0], 1'b0});
endmodule

// Carry-Save Adder Tree (4:2 based, falls back to 3:2 if needed)
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
    
    function automatic integer calc_4to2_levels(integer n);
        integer remaining = n;
        integer levels_4to2 = 0;
        while (remaining >= 4) begin
            levels_4to2 = levels_4to2 + 1;
            remaining = remaining - 2;    // 4->2 reduction
        end
        return levels_4to2;
    endfunction

    localparam LEVELS_4TO2 = calc_4to2_levels(N);
    localparam TOTAL_LEVELS = LEVELS_4TO2 + ((N - LEVELS_4TO2 * 2) == 3 ? 1 : 0);
    localparam WN = W + TOTAL_LEVELS + 2;
    
    // Intermediate signals
    wire [WN-1:0] St [0:TOTAL_LEVELS];
    wire [WN-1:0] Ct [0:TOTAL_LEVELS];
    
    // Initialize first two operands
    assign St[0] = WN'(operands[0]);
    assign Ct[0] = WN'(operands[1]);
    
    // Generate 4:2 compressor levels first
    for (genvar i = 0; i < LEVELS_4TO2; i++) begin : g_4to2_levels
        localparam WI = W + i;
        localparam WO = WI + 2;
        localparam OP_A_IDX = 2 + (i * 2);
        localparam OP_B_IDX = 3 + (i * 2);
        
        wire [WO-1:0] st, ct;
        CSA_level_4to2 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) CSA_4TO2 (
            .a     (WI'(St[i])),
            .b     (WI'(Ct[i])),
            .c     (WI'(operands[OP_A_IDX])),
            .d     (WI'(operands[OP_B_IDX])),
            .sum   (st),
            .carry (ct)
        );
        assign St[i+1] = WN'(st);
        assign Ct[i+1] = WN'(ct);
    end
    
    // If final 3:2 compressor level is needed
    if ((N - LEVELS_4TO2 * 2) == 3) begin : g_final_3to2
        // exactly 3 operands left
        localparam FINAL_OP_IDX = 2 + (LEVELS_4TO2 * 2);
        localparam WI = W + LEVELS_4TO2;
        localparam WO = WI + 2;
        
        wire [WO-1:0] st, ct;
        CSA_level_3to2 #(
            .N       (WI),
            .WIDTH_O (WO)
        ) CSA_3TO2 (
            .a     (WI'(St[LEVELS_4TO2])),
            .b     (WI'(Ct[LEVELS_4TO2])),
            .c     (WI'(operands[FINAL_OP_IDX])),
            .sum   (st),
            .carry (ct)
        );
        assign St[LEVELS_4TO2+1] = WN'(st);
        assign Ct[LEVELS_4TO2+1] = WN'(ct);
    end
    else begin : g_no_final_3to2
        // exactly 2 operands left
        if (LEVELS_4TO2 < TOTAL_LEVELS) begin : g_pass_through
            assign St[LEVELS_4TO2+1] = St[LEVELS_4TO2];
            assign Ct[LEVELS_4TO2+1] = Ct[LEVELS_4TO2];
        end
    end
    
    // Final Kogge-Stone addition
    VX_ks_adder #(
        .N (S)
    ) KSA (
        .dataa(St[TOTAL_LEVELS][S-1:0]),
        .datab(Ct[TOTAL_LEVELS][S-1:0]),
        .sum(sum),
        .cout(cout)
    );
endmodule

`TRACING_ON
