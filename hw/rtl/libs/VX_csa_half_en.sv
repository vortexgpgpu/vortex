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

// Mod-4 Carry-Save Adder Reduction Tree Structure with half enable-gated inputs
// When half_en is low, the bottom half of the input operands (except the addend, if exists) are gated off
// Set half_en to enable all inputs for summation
// Note: Only works for N = 2^(n) + {0,1} where n is an integer >= 2
// N must be 4,5,8,9,16,17,32,33,...

module VX_csa_half_en #(
    parameter N = 9,               // Number of operands
    parameter W = 8,               // Bit-width of each operand
    parameter S = W + $clog2(N)    // Output width
) (
    input  wire [N-1:0][W-1:0] operands,
    input  wire half_en,
    output wire [S-1:0] sum,
    output wire cout
);
    `STATIC_ASSERT ((N < 4 || ((N & (N-1)) != 0 || ((N-1) & (N-2)) != 0)), ("N must be of the form 2^n + {0,1} where n >= 2"));

    // Tree parameters
    localparam N_REM  = N % 4;          // 0 or 1 remainder
    localparam N_MOD4 = N - N_REM;      // Floor to multiple of 4
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
    localparam TOTAL_LEVELS = DEPTH + N_REM;
    localparam WN = W + TOTAL_LEVELS + 2;

    wire [NUM_L0-1:0][WN-1:0] level_s[0:DEPTH];
    wire [NUM_L0-1:0][WN-1:0] level_c[0:DEPTH];
    wire [NUM_L0-1:0] level_en[0:DEPTH];
    `UNUSED_VAR(level_en);
    
    // Level 0: Initial 4:2 compressors with enable gating
    for (genvar i = 0; i < NUM_L0; i = i + 1) begin : g_level0
        // Bottom half compressors are gated (i >= NUM_L0/2)
        localparam IS_BOTTOM_HALF = (i >= (NUM_L0 >> 1));
        wire comp_en = IS_BOTTOM_HALF ? half_en : 1'b1;
        
        wire [W+1:0] s_temp, c_temp;
        wire [W-1:0] gated_a, gated_b, gated_c, gated_d;
        
        // Gate inputs when disabled
        assign gated_a = comp_en ? operands[i*4 + 0] : {W{1'b0}};
        assign gated_b = comp_en ? operands[i*4 + 1] : {W{1'b0}};
        assign gated_c = comp_en ? operands[i*4 + 2] : {W{1'b0}};
        assign gated_d = comp_en ? operands[i*4 + 3] : {W{1'b0}};
        
        VX_csa_42 #(
            .N(W),
            .WIDTH_O(W+2)
        ) csa_0 (
            .a(gated_a),
            .b(gated_b),
            .c(gated_c),
            .d(gated_d),
            .sum(s_temp),
            .carry(c_temp)
        );
        assign level_s[0][i] = WN'(s_temp);
        assign level_c[0][i] = WN'(c_temp);
        assign level_en[0][i] = comp_en;
    end
    
    // Subsequent pairwise combination levels with propagated enable
    for (genvar lev = 1; lev <= DEPTH; lev = lev + 1) begin : g_levels
        localparam NUM_PREV = NUM_L0 >> (lev - 1);
        localparam NUM_CURR = (NUM_PREV + 1) >> 1;
        localparam W_IN  = W + lev * 2;
        localparam W_OUT = W_IN + 2;
        
        for (genvar i = 0; i < NUM_CURR; i = i + 1) begin : g_comps
            localparam HAS_PAIR = (i * 2 + 1) < NUM_PREV;
            
            if (HAS_PAIR) begin : g_has_pair
                // Compressor is enabled if either input pair is enabled
                wire pair_en = level_en[lev-1][i*2] | level_en[lev-1][i*2+1];
                
                wire [W_OUT-1:0] s_temp, c_temp;
                wire [W_IN-1:0] gated_s0, gated_c0, gated_s1, gated_c1;
                
                // Gate inputs based on their individual enables
                assign gated_s0 = level_en[lev-1][i*2] ? W_IN'(level_s[lev-1][i*2]) : {W_IN{1'b0}};
                assign gated_c0 = level_en[lev-1][i*2] ? W_IN'(level_c[lev-1][i*2]) : {W_IN{1'b0}};
                assign gated_s1 = level_en[lev-1][i*2+1] ? W_IN'(level_s[lev-1][i*2+1]) : {W_IN{1'b0}};
                assign gated_c1 = level_en[lev-1][i*2+1] ? W_IN'(level_c[lev-1][i*2+1]) : {W_IN{1'b0}};
                
                VX_csa_42 #(
                    .N(W_IN),
                    .WIDTH_O(W_OUT)
                ) csa_n (
                    .a(gated_s0),
                    .b(gated_c0),
                    .c(gated_s1),
                    .d(gated_c1),
                    .sum(s_temp),
                    .carry(c_temp)
                );
                assign level_s[lev][i] = WN'(s_temp);
                assign level_c[lev][i] = WN'(c_temp);
                assign level_en[lev][i] = pair_en;
            end
            else begin : g_passthrough
                // Odd one out - pass through enable as well
                assign level_s[lev][i] = level_s[lev-1][i*2];
                assign level_c[lev][i] = level_c[lev-1][i*2];
                assign level_en[lev][i] = level_en[lev-1][i*2];
            end
        end
    end
    
    wire [WN-1:0] tree_sum, tree_carry;
    `UNUSED_VAR({tree_sum, tree_carry});
    if (DEPTH == 0) begin : g_depth0
        assign tree_sum   = level_s[0][0];
        assign tree_carry = level_c[0][0];
    end
    else begin : g_depth_n
        assign tree_sum = level_s[DEPTH][0];
        assign tree_carry = level_c[DEPTH][0];
    end

    localparam W_FINAL_IN  = W + DEPTH * 2 + 2;
    localparam W_FINAL_OUT = W_FINAL_IN + 2;
    wire [W_FINAL_OUT-1:0] final_sum, final_carry;
    `UNUSED_VAR({final_sum, final_carry});    
    
    // Handle addend/remainder operand if exists
    if (N_REM == 1) begin : g_has_rem
        VX_csa_32 #(
            .N(W_FINAL_IN),
            .WIDTH_O(W_FINAL_OUT)
        ) csa_rem (
            .a(W_FINAL_IN'(tree_sum)),
            .b(W_FINAL_IN'(tree_carry)),
            .c(W_FINAL_IN'(operands[N-1])),
            .sum(final_sum),
            .carry(final_carry)
        );
    end
    else begin : g_no_rem
        // No remainder operand, pass through
        assign final_sum   = W_FINAL_OUT'(tree_sum);
        assign final_carry = W_FINAL_OUT'(tree_carry);
    end

    // Final Kogge-Stone Adder
    VX_ks_adder #(
        .N(S)
    ) ksa (
        .dataa(S'(final_sum)),
        .datab(S'(final_carry)),
        .sum(sum),
        .cout(cout)
    );
endmodule

`TRACING_ON

