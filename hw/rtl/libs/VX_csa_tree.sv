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
// Linear reduction module
module VX_csa_block #(
    parameter N = 4,
    parameter W = 8,
    parameter S = W + $clog2(N)
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [S-1:0]        sum,
    output wire [S-1:0]        carry
);
    `STATIC_ASSERT (N >= 2, ("N must be at least 2"));

    function automatic integer calc_4to2_levels(integer n);
        integer remaining = n;
        integer levels_4to2 = 0;
        while (remaining >= 4) begin
            levels_4to2 = levels_4to2 + 1;
            remaining = remaining - 2;
        end
        return levels_4to2;
    endfunction

    localparam LEVELS_4TO2 = calc_4to2_levels(N);
    localparam TOTAL_LEVELS = LEVELS_4TO2 + ((N - LEVELS_4TO2 * 2) == 3 ? 1 : 0);

    localparam WN_CALC = W + TOTAL_LEVELS + 2;
    localparam WN = (WN_CALC > S) ? WN_CALC : S;

    wire [WN-1:0] St [0:TOTAL_LEVELS];
    wire [WN-1:0] Ct [0:TOTAL_LEVELS];

    assign St[0] = WN'(operands[0]);
    assign Ct[0] = WN'(operands[1]);

    for (genvar i = 0; i < LEVELS_4TO2; i++) begin : g_4to2_levels
        localparam WI = W + i;
        localparam WO = WI + 2;
        localparam OP_A_IDX = 2 + (i * 2);
        localparam OP_B_IDX = 3 + (i * 2);

        wire [WO-1:0] st, ct;

        VX_csa_42 #(
            .N(WI),
            .WIDTH_O(WO)
        ) csa_42 (
            .a(WI'(St[i])),
            .b(WI'(Ct[i])),
            .c(WI'(operands[OP_A_IDX])),
            .d(WI'(operands[OP_B_IDX])),
            .sum(st),
            .carry(ct)
        );
        assign St[i+1] = WN'(st);
        assign Ct[i+1] = WN'(ct);
    end

    if ((N - LEVELS_4TO2 * 2) == 3) begin : g_final_3to2
        localparam FINAL_OP_IDX = 2 + (LEVELS_4TO2 * 2);
        localparam WI = W + LEVELS_4TO2;
        localparam WO = WI + 2;

        wire [WO-1:0] st, ct;

        VX_csa_32 #(
            .N(WI),
            .WIDTH_O(WO)
        ) csa_32 (
            .a(WI'(St[LEVELS_4TO2])),
            .b(WI'(Ct[LEVELS_4TO2])),
            .c(WI'(operands[FINAL_OP_IDX])),
            .sum(st),
            .carry(ct)
        );
        assign St[LEVELS_4TO2+1] = WN'(st);
        assign Ct[LEVELS_4TO2+1] = WN'(ct);
    end
    else begin : g_no_final_3to2
        if (LEVELS_4TO2 < TOTAL_LEVELS) begin : g_pass_through
            assign St[LEVELS_4TO2+1] = St[LEVELS_4TO2];
            assign Ct[LEVELS_4TO2+1] = Ct[LEVELS_4TO2];
        end
    end

    assign sum  = St[TOTAL_LEVELS][S-1:0];
    assign carry = Ct[TOTAL_LEVELS][S-1:0];
endmodule

// Tree reduction module
module VX_csa_tree #(
    parameter N   = 16, // Number of operands
    parameter W   = 8,  // Bit-width of each operand
    parameter K   = 6,  // Cluster Size (1 = packed (ASIC), >1 = clustered (FPGA))
    parameter BAL = 1,  // Balancing Mode (0 = Ragged, 1 = Balanced)
    parameter S   = W + $clog2(N)
) (
    input  wire [N-1:0][W-1:0] operands,
    output wire [S-1:0]        sum,
    output wire [S-1:0]        carry
);

    localparam CLUSTER_W    = (K == 1) ? W : (W + K + 2);
    localparam NUM_CLUSTERS = (K == 1) ? N : ((N + K - 1) / K);
    localparam TOP_N        = (K == 1) ? N : (NUM_CLUSTERS * 2);

    // --- Helper Functions ---
    function automatic integer next_lev_ragged(integer n_in);
        integer n_rem = n_in % 4;
        return (n_rem == 3) ? ((n_in/4)*2 + 2) : ((n_in/4)*2 + n_rem);
    endfunction

    function automatic integer next_lev_balanced(integer n_in);
        integer n_rem = n_in % 4;
        if (n_in <= 3) return (n_rem == 3) ? 2 : n_rem;
        if (n_rem == 0) return (n_in/4) * 2;
        if (n_rem == 1 || n_rem == 2) return ((n_in/4)-1)*2 + 4;
        return ((n_in/4)*2) + 2;
    endfunction

    function automatic integer get_next_sz(integer n, integer use_bal);
        return (use_bal != 0) ? next_lev_balanced(n) : next_lev_ragged(n);
    endfunction

    function automatic integer calc_depth(integer n_in);
        integer d = 0;
        integer count = n_in;
        while (count > 2) begin
            count = get_next_sz(count, BAL);
            d = d + 1;
        end
        return d;
    endfunction

    function automatic integer get_cnt_at_lev(integer l, integer start_n);
        integer c = start_n;
        for(integer k=0; k<l; k++) c = get_next_sz(c, BAL);
        return c;
    endfunction

    // --- Input Clustering ---

    wire [TOP_N-1:0][CLUSTER_W-1:0] tree_inputs;

    if (K == 1) begin : g_no_cluster
        for (genvar i = 0; i < N; ++i) begin : g_map
            assign tree_inputs[i] = CLUSTER_W'(operands[i]);
        end
    end else begin : g_do_cluster
        for (genvar i = 0; i < NUM_CLUSTERS; ++i) begin : g_clusters
            localparam C_SIZE = (i == NUM_CLUSTERS - 1) ? (N - i*K) : K;

            if (C_SIZE == 1) begin : g_c1
                assign tree_inputs[2*i]     = CLUSTER_W'(operands[i*K]);
                assign tree_inputs[2*i + 1] = '0;
            end else begin : g_lin
                wire [CLUSTER_W-1:0] s_loc, c_loc;
                wire [C_SIZE-1:0][W-1:0] sub_ops;

                for(genvar j=0; j<C_SIZE; ++j) begin : g_slice
                    assign sub_ops[j] = operands[i*K + j];
                end

                VX_csa_block #(
                    .N(C_SIZE),
                    .W(W),
                    .S(CLUSTER_W)
                ) cluster_linear (
                    .operands(sub_ops),
                    .sum(s_loc),
                    .carry(c_loc)
                );

                assign tree_inputs[2*i]     = s_loc;
                assign tree_inputs[2*i + 1] = c_loc;
            end
        end
    end

    // --- Tree Generation ---

    localparam DEPTH = calc_depth(TOP_N);
    localparam MAX_WN = CLUSTER_W + (DEPTH * 2) + 4;
    localparam WN = (S < MAX_WN) ? S : MAX_WN;

    wire [WN-1:0] tree_sig [DEPTH+1][TOP_N] /* verilator split_var */;

    for (genvar i = 0; i < TOP_N; i++) begin : g_init_l0
        assign tree_sig[0][i] = WN'(tree_inputs[i]);
    end

    for (genvar lev = 0; lev < DEPTH; lev++) begin : g_levels
        localparam integer NUM_IN = get_cnt_at_lev(lev, TOP_N);
        localparam integer WI = CLUSTER_W + (lev * 2);
        localparam integer WO = WI + 2;

        if (BAL == 1) begin : g_balanced
            localparam integer N_42 = NUM_IN / 4;
            localparam integer REM  = NUM_IN % 4;
            localparam integer CORE = (REM == 1 || REM == 2) ? (N_42 - 1) : N_42;

            for (genvar i = 0; i < CORE; i++) begin : g_core
                wire [WO-1:0] s, c;
                `UNUSED_VAR ({s, c})
                VX_csa_42 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) csa (
                    .a(WI'(tree_sig[lev][i*4+0])),
                    .b(WI'(tree_sig[lev][i*4+1])),
                    .c(WI'(tree_sig[lev][i*4+2])),
                    .d(WI'(tree_sig[lev][i*4+3])),
                    .sum(s),
                    .carry(c)
                );
                assign tree_sig[lev+1][i*2+0] = WN'(s);
                assign tree_sig[lev+1][i*2+1] = WN'(c);
            end

            if (REM == 1) begin : g_r1
                localparam B = CORE*4; localparam OB = CORE*2;
                wire [WO-1:0] s, c;
                VX_csa_32 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) c32 (
                    .a(WI'(tree_sig[lev][B])),
                    .b(WI'(tree_sig[lev][B+1])),
                    .c(WI'(tree_sig[lev][B+2])),
                    .sum(s),
                    .carry(c)
                );
                assign tree_sig[lev+1][OB+0] = WN'(s); assign tree_sig[lev+1][OB+1] = WN'(c);
                assign tree_sig[lev+1][OB+2] = tree_sig[lev][B+3]; assign tree_sig[lev+1][OB+3] = tree_sig[lev][B+4];
            end
            else if (REM == 2) begin : g_r2
                localparam B = CORE*4; localparam OB = CORE*2;
                wire [WO-1:0] s1, c1, s2, c2;
                VX_csa_32 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) cA (
                    .a(WI'(tree_sig[lev][B])),
                    .b(WI'(tree_sig[lev][B+1])),
                    .c(WI'(tree_sig[lev][B+2])),
                    .sum(s1),
                    .carry(c1)
                );
                VX_csa_32 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) cB (
                    .a(WI'(tree_sig[lev][B+3])),
                    .b(WI'(tree_sig[lev][B+4])),
                    .c(WI'(tree_sig[lev][B+5])),
                    .sum(s2),
                    .carry(c2)
                );
                assign tree_sig[lev+1][OB+0] = WN'(s1); assign tree_sig[lev+1][OB+1] = WN'(c1);
                assign tree_sig[lev+1][OB+2] = WN'(s2); assign tree_sig[lev+1][OB+3] = WN'(c2);
            end
            else if (REM == 3) begin : g_r3
                localparam B = CORE*4; localparam OB = CORE*2;
                wire [WO-1:0] s, c;
                VX_csa_32 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) c32 (
                    .a(WI'(tree_sig[lev][B])),
                    .b(WI'(tree_sig[lev][B+1])),
                    .c(WI'(tree_sig[lev][B+2])),
                    .sum(s),
                    .carry(c)
                );
                assign tree_sig[lev+1][OB+0] = WN'(s); assign tree_sig[lev+1][OB+1] = WN'(c);
            end
        end
        else begin : g_ragged
            localparam integer N_42 = NUM_IN / 4;
            localparam integer REM  = NUM_IN % 4;
            for (genvar i = 0; i < N_42; i++) begin : g_csa42
                wire [WO-1:0] s, c;
                VX_csa_42 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) csa (
                    .a(WI'(tree_sig[lev][i*4+0])),
                    .b(WI'(tree_sig[lev][i*4+1])),
                    .c(WI'(tree_sig[lev][i*4+2])),
                    .d(WI'(tree_sig[lev][i*4+3])),
                    .sum(s),
                    .carry(c)
                );
                assign tree_sig[lev+1][i*2+0] = WN'(s);
                assign tree_sig[lev+1][i*2+1] = WN'(c);
            end
            if (REM == 3) begin : g_rem3
                wire [WO-1:0] s, c;
                VX_csa_32 #(
                    .N(WI),
                    .WIDTH_O(WO)
                ) c32 (
                    .a(WI'(tree_sig[lev][N_42*4])),
                    .b(WI'(tree_sig[lev][N_42*4+1])),
                    .c(WI'(tree_sig[lev][N_42*4+2])),
                    .sum(s),
                    .carry(c)
                );
                assign tree_sig[lev+1][N_42*2] = WN'(s);
                assign tree_sig[lev+1][N_42*2+1] = WN'(c);
            end
            else begin : g_pass
                for(genvar p=0; p<REM; p++) begin : g_rem
                    assign tree_sig[lev+1][N_42*2+p] = tree_sig[lev][N_42*4+p];
                end
            end
        end
    end

    assign sum   = S'(tree_sig[DEPTH][0]);
    assign carry = (get_cnt_at_lev(DEPTH, TOP_N) > 1) ? S'(tree_sig[DEPTH][1]) : '0;

endmodule
`TRACING_ON
