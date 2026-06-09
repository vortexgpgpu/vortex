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

// Greedy-Then-Oldest (GTO) arbiter.
//
// Greedy: keep granting the same requester as long as it is active.
// Oldest: when the current requester deasserts, grant the least-recently-
//         granted requester.
//
// Selection priority is held in a dominance matrix: the granted requester is
// demoted below all others, so "least recently granted" approximates "oldest"
// and selection has constant logic depth (a one-hot dominance check) rather
// than a serial age-counter max-reduction. Initial order is by index.

`TRACING_OFF
module VX_gto_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,
    input  wire [NUM_REQS-1:0]      suppress,
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,
    output wire                     grant_valid,
    input  wire                     grant_ready
);
    if (NUM_REQS == 1) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (suppress)
        `UNUSED_VAR (grant_ready)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0] && ~suppress[0];

    end else begin : g_arbiter

        // Effective requests: eligible for selection (not suppressed).
        // Age tracking still uses `requests` so suppressed warps keep aging.
        wire [NUM_REQS-1:0] eff_requests = requests & ~suppress;

        // -- Greedy: sticky grant ------------------------------------------

        reg [NUM_REQS-1:0] prev_grant;
        always @(posedge clk) begin
            if (reset) begin
                prev_grant <= '0;
            end else if (grant_valid && grant_ready) begin
                prev_grant <= grant_onehot;
            end
        end

        wire retain_grant = |(prev_grant & eff_requests);

        wire grant_fire = grant_valid && grant_ready;

        // -- Oldest: least-recently-granted via a priority matrix ----------
        // rank_matrix[i][j] (upper triangle, i<j) = 1 means requester i
        // outranks j. The reconstructed `outranks` relation is antisymmetric.
        // On each grant the winner is demoted below every other requester,
        // which preserves a valid total order, so `oldest_onehot` is always
        // exactly one-hot. This replaces the serial age-counter max-reduction
        // with a constant-depth dominance check (one AND-fanin level over the
        // requesters). The matrix moves only on grant, so a suppressed-but-
        // pending requester keeps its rank and wins once unsuppressed.

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
                assign dominates[j] = (i == j) ? 1'b1 : (outranks[i][j] | ~eff_requests[j]);
            end
            assign oldest_onehot[i] = eff_requests[i] && (&dominates);
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

        // Greedy wins if the current grantee is still active; otherwise the
        // matrix-selected oldest. prev_grant is already one-hot, so the greedy
        // path needs no encoder.
        assign grant_valid  = |eff_requests;
        assign grant_onehot = retain_grant ? prev_grant : oldest_onehot;

        // grant_onehot is one-hot (or zero when idle, gated by grant_valid), so
        // grant_index is a flat per-bit OR-reduce: exact for a one-hot input,
        // with no priority cascade and no separate encoders.
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
