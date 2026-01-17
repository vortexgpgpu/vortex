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

// Mod-4 Carry-Save Adder Reduction Tree Structure for large operand counts
module VX_csa_mod4 #(
    parameter N = 11,              // Number of operands (N >= 3)
    parameter W = 8,               // Bit-width of each operand
    parameter S = W + $clog2(N)    // Output width
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [S-1:0] sum,
    output wire cout
);
    `STATIC_ASSERT (N >= 7, ("N must at least be 7"));

    // Tree parameters
    localparam N_MOD4 = (N >> 2) << 2;  // Floor to multiple of 4
    localparam N_REM  = N - N_MOD4;     // Remainder: 0, 1, 2, or 3
    localparam NUM_L0 = N_MOD4 >> 2;    // Number of initial 4:2 compressors
    
    // Calculate tree depth (number of levels of 4:2)
    function automatic integer calc_depth(integer n);
        integer d;
        d = 0;
        while (n > 1) begin
            d = d + 1;
            n = (n + 1) >> 1;
        end
        return d;
    endfunction
    
    localparam DEPTH = calc_depth(NUM_L0);
    localparam TOTAL_LEVELS = DEPTH + ((N_REM == 3) ? 2 : (N_REM > 0) ? 1 : 0);
    localparam WN = W + TOTAL_LEVELS + 2;

    // Build parallel mod-4 tree
    wire [NUM_L0-1:0][WN-1:0] level_s[0:DEPTH];
    wire [NUM_L0-1:0][WN-1:0] level_c[0:DEPTH];
    
    // Level 0: Initial 4:2 compressors
    for (genvar i = 0; i < NUM_L0; i = i + 1) begin : g_level0
        wire [W+1:0] s_temp, c_temp;
        VX_csa_42 #(
            .N(W),
            .WIDTH_O(W+2)
        ) csa_0 (
            .a(operands[i*4 + 0]),
            .b(operands[i*4 + 1]),
            .c(operands[i*4 + 2]),
            .d(operands[i*4 + 3]),
            .sum(s_temp),
            .carry(c_temp)
        );
        assign level_s[0][i] = WN'(s_temp);
        assign level_c[0][i] = WN'(c_temp);
    end
    
    // Subsequent pairwise combination levels
    for (genvar lev = 1; lev <= DEPTH; lev = lev + 1) begin : g_levels
        localparam NUM_PREV = NUM_L0 >> (lev - 1);
        localparam NUM_CURR = (NUM_PREV + 1) >> 1;
        localparam W_IN  = W + lev * 2;
        localparam W_OUT = W_IN + 2;
        
        for (genvar i = 0; i < NUM_CURR; i = i + 1) begin : g_comps
            localparam HAS_PAIR = (i * 2 + 1) < NUM_PREV;
            
            if (HAS_PAIR) begin : g_has_pair
`IGNORE_UNOPTFLAT_BEGIN
                wire [W_OUT-1:0] s_temp, c_temp;
                `UNUSED_VAR({s_temp, c_temp});
`IGNORE_UNOPTFLAT_END
                VX_csa_42 #(
                    .N(W_IN),
                    .WIDTH_O(W_OUT)
                ) csa_n (
                    .a(W_IN'(level_s[lev-1][i*2])),
                    .b(W_IN'(level_c[lev-1][i*2])),
                    .c(W_IN'(level_s[lev-1][i*2+1])),
                    .d(W_IN'(level_c[lev-1][i*2+1])),
                    .sum(s_temp),
                    .carry(c_temp)
                );
                assign level_s[lev][i] = WN'(s_temp);
                assign level_c[lev][i] = WN'(c_temp);
            end
            else begin : g_passthrough
                assign level_s[lev][i] = level_s[lev-1][i*2];
                assign level_c[lev][i] = level_c[lev-1][i*2];
            end
        end
    end
    
    wire [WN-1:0] tree_sum, tree_carry;
    assign tree_sum   = level_s[DEPTH][0];
    assign tree_carry = level_c[DEPTH][0];
    `UNUSED_VAR({tree_sum, tree_carry});

    wire [S-1:0] ksa_sum, ksa_carry;
    
    // Final reduction based on remainder
    if (N_REM == 0) begin : g_rem_0
        assign ksa_sum   = S'(tree_sum);
        assign ksa_carry = S'(tree_carry);
    end
    else if (N_REM == 1) begin : g_rem_1
        localparam W_F1_IN  = W + DEPTH * 2 + 2;
        localparam W_F1_OUT = W_F1_IN + 2;
        wire [W_F1_OUT-1:0] f1_s, f1_c;
        `UNUSED_VAR({f1_s, f1_c});
        
        VX_csa_32 #(
            .N(W_F1_IN),
            .WIDTH_O(W_F1_OUT)
        ) csa_rem1 (
            .a(W_F1_IN'(tree_sum)),
            .b(W_F1_IN'(tree_carry)),
            .c(W_F1_IN'(operands[N-1])),
            .sum(f1_s),
            .carry(f1_c)
        );
        
        assign ksa_sum   = S'(f1_s);
        assign ksa_carry = S'(f1_c);
    end
    else if (N_REM == 2) begin : g_rem_2
        localparam W_F2_IN  = W + DEPTH * 2 + 2;
        localparam W_F2_OUT = W_F2_IN + 2;
        wire [W_F2_OUT-1:0] f2_s, f2_c;
        `UNUSED_VAR({f2_s, f2_c});
        
        VX_csa_42 #(
            .N(W_F2_IN),
            .WIDTH_O(W_F2_OUT)
        ) csa_rem2 (
            .a(W_F2_IN'(tree_sum)),
            .b(W_F2_IN'(tree_carry)),
            .c(W_F2_IN'(operands[N-2])),
            .d(W_F2_IN'(operands[N-1])),
            .sum(f2_s),
            .carry(f2_c)
        );
        
        assign ksa_sum   = S'(f2_s);
        assign ksa_carry = S'(f2_c);
    end
    else begin : g_rem_3
        localparam W_F3A_IN  = W + DEPTH * 2 + 2;
        localparam W_F3A_OUT = W_F3A_IN + 2;
        wire [W_F3A_OUT-1:0] f3a_s, f3a_c;
        
        // First 3:2: tree_sum + tree_carry + 1 remaining operand
        VX_csa_32 #(
            .N(W_F3A_IN),
            .WIDTH_O(W_F3A_OUT)
        ) csa_rem3a (
            .a(W_F3A_IN'(tree_sum)),
            .b(W_F3A_IN'(tree_carry)),
            .c(W_F3A_IN'(operands[N-3])),
            .sum(f3a_s),
            .carry(f3a_c)
        );
        
        // Final 4:2: f3a outputs + remaining 2 operands
        localparam W_F3B_IN  = W_F3A_OUT;
        localparam W_F3B_OUT = W_F3B_IN + 2;
        wire [W_F3B_OUT-1:0] f3b_s, f3b_c;
        `UNUSED_VAR({f3b_s, f3b_c});
        
        VX_csa_42 #(
            .N(W_F3B_IN),
            .WIDTH_O(W_F3B_OUT)
        ) csa_rem3b (
            .a(f3a_s),
            .b(f3a_c),
            .c(W_F3B_IN'(operands[N-2])),
            .d(W_F3B_IN'(operands[N-1])),
            .sum(f3b_s),
            .carry(f3b_c)
        );
        
        assign ksa_sum   = S'(f3b_s);
        assign ksa_carry = S'(f3b_c);
    end

    VX_ks_adder #(
        .N(S)
    ) ksa (
        .dataa(ksa_sum),
        .datab(ksa_carry),
        .sum(sum),
        .cout(cout)
    );
endmodule

`TRACING_ON

