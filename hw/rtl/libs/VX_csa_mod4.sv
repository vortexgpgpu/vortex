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
    parameter N = 11,             // Number of operands (N >= 3)
    parameter W = 8,              // Bit-width of each operand
    parameter S = W + $clog2(N)   // Output width
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [S-1:0] sum,
    output wire cout
);
    `STATIC_ASSERT (N >= 7, ("N must at least be 7"));

    // Tree parameters
    localparam N_MOD4 = (N >> 2) << 2; // Floor to multiple of 4
    localparam N_REM  = N - N_MOD4;    // Remainder: 0, 1, 2, or 3
    localparam NUM_L0 = N_MOD4 >> 2;   // Number of initial 4:2 compressors

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

    // 1. Calculate Max Width (Tree grows by 2 bits per level)
    localparam MAX_WN = W + TOTAL_LEVELS * 2 + 2;

    // 2. Clamp Internal Width to S (Area Optimization)
    localparam WN = (S < MAX_WN) ? S : MAX_WN;

    // Build parallel mod-4 tree
    wire [NUM_L0-1:0][WN-1:0] level_s[0:DEPTH];
    wire [NUM_L0-1:0][WN-1:0] level_c[0:DEPTH];

    // -------------------------------------------------------------------------
    // Level 0: Initial 4:2 compressors
    // -------------------------------------------------------------------------

    for (genvar i = 0; i < NUM_L0; i = i + 1) begin : g_level0
        localparam _WI = W;
        localparam _WO = _WI + 2;
        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        wire [WO-1:0] s_temp, c_temp;

        // Cast inputs to WI (Sign-Extend if Signed)
        wire [WI-1:0] op_a = WI'(operands[i*4 + 0]);
        wire [WI-1:0] op_b = WI'(operands[i*4 + 1]);
        wire [WI-1:0] op_c = WI'(operands[i*4 + 2]);
        wire [WI-1:0] op_d = WI'(operands[i*4 + 3]);

        VX_csa_42 #(
            .N(WI),
            .WIDTH_O(WO)
        ) csa_0 (
            .a(op_a),
            .b(op_b),
            .c(op_c),
            .d(op_d),
            .sum(s_temp),
            .carry(c_temp)
        );

        assign level_s[0][i] = WN'(s_temp);
        assign level_c[0][i] = WN'(c_temp);
    end

    // -------------------------------------------------------------------------
    // Subsequent pairwise combination levels
    // -------------------------------------------------------------------------

    for (genvar lev = 1; lev <= DEPTH; lev = lev + 1) begin : g_levels
        localparam NUM_PREV = NUM_L0 >> (lev - 1);
        localparam NUM_CURR = (NUM_PREV + 1) >> 1;

        localparam _WI = W + (lev * 2);
        localparam _WO = _WI + 2;
        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        for (genvar i = 0; i < NUM_CURR; i = i + 1) begin : g_comps
            localparam HAS_PAIR = (i * 2 + 1) < NUM_PREV;
            if (HAS_PAIR) begin : g_has_pair
                wire [WO-1:0] s_temp, c_temp;

                // Cast inputs to WI (Sign-Extend if Signed)
                wire [WI-1:0] op_a = WI'(level_s[lev-1][i*2]);
                wire [WI-1:0] op_b = WI'(level_c[lev-1][i*2]);
                wire [WI-1:0] op_c = WI'(level_s[lev-1][i*2+1]);
                wire [WI-1:0] op_d = WI'(level_c[lev-1][i*2+1]);

                VX_csa_42 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) csa_n (
                    .a(op_a),
                    .b(op_b),
                    .c(op_c),
                    .d(op_d),
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

    wire [WN-1:0] tree_sum_w   = level_s[DEPTH][0];
    wire [WN-1:0] tree_carry_w = level_c[DEPTH][0];

    // -------------------------------------------------------------------------
    // Final reduction based on remainder
    // -------------------------------------------------------------------------

    wire [WN-1:0] final_sum, final_carry;

    if (N_REM == 0) begin : g_rem_0
        assign final_sum   = tree_sum_w;
        assign final_carry = tree_carry_w;
    end
    else if (N_REM == 1) begin : g_rem_1
        localparam _WI = W + (DEPTH * 2) + 2;
        localparam _WO = _WI + 2;

        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        wire [WO-1:0] f1_s, f1_c;

        wire [WI-1:0] op_a = WI'(tree_sum_w);
        wire [WI-1:0] op_b = WI'(tree_carry_w);
        wire [WI-1:0] op_c = WI'(operands[N-1]);

        VX_csa_32 #(
            .N(WI),
            .WIDTH_O(WO)
        ) csa_rem1 (
            .a(op_a),
            .b(op_b),
            .c(op_c),
            .sum(f1_s),
            .carry(f1_c)
        );

        assign final_sum   = WN'(f1_s);
        assign final_carry = WN'(f1_c);
    end
    else if (N_REM == 2) begin : g_rem_2
        localparam _WI = W + (DEPTH * 2) + 2;
        localparam _WO = _WI + 2;
        localparam WI = (_WI > WN) ? WN : _WI;
        localparam WO = (_WO > WN) ? WN : _WO;

        wire [WO-1:0] f2_s, f2_c;

        // Cast inputs to WI (Sign-Extend if Signed)
        wire [WI-1:0] op_a = WI'(tree_sum_w);
        wire [WI-1:0] op_b = WI'(tree_carry_w);
        wire [WI-1:0] op_c = WI'(operands[N-2]);
        wire [WI-1:0] op_d = WI'(operands[N-1]);

        VX_csa_42 #(
            .N(WI),
            .WIDTH_O(WO)
        ) csa_rem2 (
            .a(op_a),
            .b(op_b),
            .c(op_c),
            .d(op_d),
            .sum(f2_s),
            .carry(f2_c)
        );

        assign final_sum   = WN'(f2_s);
        assign final_carry = WN'(f2_c);
    end
    else begin : g_rem_3
        // STEP 1: 3:2 CSA
        localparam _WI_A = W + (DEPTH * 2) + 2;
        localparam _WO_A = _WI_A + 2;
        localparam WI_A = (_WI_A > WN) ? WN : _WI_A;
        localparam WO_A = (_WO_A > WN) ? WN : _WO_A;

        // Cast inputs to WI (Sign-Extend if Signed)
        wire [WI_A-1:0] op_a1 = WI_A'(tree_sum_w);
        wire [WI_A-1:0] op_b1 = WI_A'(tree_carry_w);
        wire [WI_A-1:0] op_c1 = WI_A'(operands[N-3]);

        wire [WO_A-1:0] f3a_s, f3a_c;

        VX_csa_32 #(
            .N(WI_A),
            .WIDTH_O(WO_A)
        ) csa_rem3a (
            .a(op_a1),
            .b(op_b1),
            .c(op_c1),
            .sum(f3a_s),
            .carry(f3a_c)
        );

        // STEP 2: 4:2 CSA
        localparam _WI_B = _WO_A;
        localparam _WO_B = _WI_B + 2;
        localparam WI_B = (_WI_B > WN) ? WN : _WI_B;
        localparam WO_B = (_WO_B > WN) ? WN : _WO_B;

        // Cast inputs to WI (Sign-Extend if Signed)
        wire [WI_B-1:0] op_a2 = WI_B'(f3a_s);
        wire [WI_B-1:0] op_b2 = WI_B'(f3a_c);
        wire [WI_B-1:0] op_c2 = WI_B'(operands[N-2]);
        wire [WI_B-1:0] op_d2 = WI_B'(operands[N-1]);

        wire [WO_B-1:0] f3b_s, f3b_c;

        VX_csa_42 #(
            .N(WI_B),
            .WIDTH_O(WO_B)
        ) csa_rem3b (
            .a(op_a2),
            .b(op_b2),
            .c(op_c2),
            .d(op_d2),
            .sum(f3b_s),
            .carry(f3b_c)
        );

        assign final_sum   = WN'(f3b_s);
        assign final_carry = WN'(f3b_c);
    end

    // -------------------------------------------------------------------------
    // Final KS Adder
    // -------------------------------------------------------------------------

    wire [WN-1:0] raw_sum;

    VX_ks_adder #(
        .N(WN)
    ) ksa (
        .cin(0),
        .dataa(final_sum),
        .datab(final_carry),
        .sum(raw_sum),
        .cout(cout)
    );

    // Final Output Expansion
    assign sum = S'(raw_sum);

endmodule

`TRACING_ON
