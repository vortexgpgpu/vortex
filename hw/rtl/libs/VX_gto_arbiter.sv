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

`include "VX_platform.vh"

// Greedy-Then-Oldest (GTO) arbiter: hold the current grantee while active,
// else pick the least-recently-granted requester. Order is held in a dominance
// matrix (granted requester demoted below all others), giving constant-depth
// selection instead of a serial age-counter max-reduction. Initial order by index.

`TRACING_OFF
module VX_gto_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,
    output wire                     grant_valid,
    input  wire                     grant_ready
);
    if (NUM_REQS == 1) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (grant_ready)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin : g_arbiter

        // -- Greedy: sticky grant ------------------------------------------

        reg [NUM_REQS-1:0] prev_grant;
        always @(posedge clk) begin
            if (reset) begin
                prev_grant <= '0;
            end else if (grant_valid && grant_ready) begin
                prev_grant <= grant_onehot;
            end
        end

        wire retain_grant = |(prev_grant & requests);

        wire grant_fire = grant_valid && grant_ready;

        // -- Oldest: least-recently-granted via a dominance matrix ---------
        // rank_matrix[i][j] (i<j) = 1 means i outranks j; `outranks` reconstructs
        // the full antisymmetric relation. Demoting the winner below all others
        // keeps a valid total order, so `oldest_onehot` is exactly one-hot. The
        // matrix moves only on grant, so a pending requester keeps its rank.

        reg [NUM_REQS-1:0] rank_matrix [NUM_REQS];

        wire [NUM_REQS-1:0] outranks [NUM_REQS];
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_outranks
            for (genvar j = 0; j < NUM_REQS; ++j) begin : g_col
                assign outranks[i][j] = (i < j) ? rank_matrix[i][j]
                                      : (i > j) ? ~rank_matrix[j][i]
                                      :           1'b0;
            end
        end

        wire [NUM_REQS-1:0] oldest_onehot;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_oldest
            wire [NUM_REQS-1:0] dominates;
            for (genvar j = 0; j < NUM_REQS; ++j) begin : g_dom
                // i is not blocked by j when it outranks j or j is not requesting.
                assign dominates[j] = (i == j) ? 1'b1 : (outranks[i][j] | ~requests[j]);
            end
            assign oldest_onehot[i] = requests[i] && (&dominates);
        end

        integer mr, mc;
        always @(posedge clk) begin
            if (reset) begin
                for (mr = 0; mr < NUM_REQS; mr = mr + 1)
                    for (mc = 0; mc < NUM_REQS; mc = mc + 1)
                        if (mr < mc) rank_matrix[mr][mc] <= 1'b1; // low index outranks high (tie-break)
            end else if (grant_fire) begin
                for (mr = 0; mr < NUM_REQS; mr = mr + 1)
                    for (mc = 0; mc < NUM_REQS; mc = mc + 1)
                        if (mr < mc) begin
                            if      (grant_onehot[mr]) rank_matrix[mr][mc] <= 1'b0; // mr granted -> demote
                            else if (grant_onehot[mc]) rank_matrix[mr][mc] <= 1'b1; // mc granted -> mr rises
                        end
            end
        end

        // Greedy wins if the current grantee is still active, else the oldest.
        // prev_grant is already one-hot, so the greedy path needs no encoder.
        assign grant_valid  = |requests;
        assign grant_onehot = retain_grant ? prev_grant : oldest_onehot;

        // grant_onehot is one-hot (zero when idle, gated by grant_valid), so
        // grant_index is a flat per-bit OR-reduce with no priority cascade.
        for (genvar k = 0; k < LOG_NUM_REQS; ++k) begin : g_grant_index
            wire [NUM_REQS-1:0] kbits;
            for (genvar i = 0; i < NUM_REQS; ++i) begin : g_kbit
                assign kbits[i] = (((i >> k) & 1) != 0) ? grant_onehot[i] : 1'b0;
            end
            assign grant_index[k] = |kbits;
        end

    end

endmodule
`TRACING_ON
