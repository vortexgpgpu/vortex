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
// Oldest: when the current requester deasserts, grant the requester
//         that has been waiting the longest (earliest arrival).
//
// Per-requester age is tracked with a saturating timestamp counter
// (`AGE_W` bits, default 4 → 16 distinct ages).  The counter increments
// every cycle the request is asserted and has not been granted.  It
// resets when the request is granted or deasserted.
//
// When multiple requesters share the maximum age the lowest index wins
// (tie-break via priority encoder on the age-match mask).

`TRACING_OFF
module VX_gto_arbiter #(
    parameter NUM_REQS     = 1,
    parameter AGE_W        = `LOG2UP(NUM_REQS),
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

        // -- Per-requester age counters ------------------------------------

        reg [NUM_REQS-1:0][AGE_W-1:0] age_r;
        wire grant_fire = grant_valid && grant_ready;

        always @(posedge clk) begin
            if (reset) begin
                for (int i = 0; i < NUM_REQS; ++i)
                    age_r[i] <= '0;
            end else begin
                for (int i = 0; i < NUM_REQS; ++i) begin
                    if (!requests[i] || (grant_fire && grant_onehot[i])) begin
                        // Request deasserted or just granted: reset age.
                        age_r[i] <= '0;
                    end else begin
                        // Waiting: increment age (saturate at max).
                        if (age_r[i] != {AGE_W{1'b1}})
                            age_r[i] <= age_r[i] + AGE_W'(1);
                    end
                end
            end
        end

        // -- Find oldest requester -----------------------------------------
        // Compute the maximum age among active requesters, then build a mask
        // of requesters at that maximum age.  Priority-encode the mask to
        // break ties (lowest index wins).

        reg [AGE_W-1:0] max_age;
        always @(*) begin
            max_age = '0;
            for (int i = 0; i < NUM_REQS; ++i) begin
                if (requests[i] && (age_r[i] > max_age))
                    max_age = age_r[i];
            end
        end

        wire [NUM_REQS-1:0] oldest_mask;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_oldest
            assign oldest_mask[i] = requests[i] && (age_r[i] == max_age);
        end

        wire [LOG_NUM_REQS-1:0] oldest_index;
        wire [NUM_REQS-1:0]     oldest_onehot;
        wire                    oldest_valid;

        VX_priority_encoder #(
            .N (NUM_REQS)
        ) oldest_sel (
            .data_in    (oldest_mask),
            .index_out  (oldest_index),
            .onehot_out (oldest_onehot),
            .valid_out  (oldest_valid)
        );

        // -- Output mux: greedy wins if current requester is still active --

        wire [LOG_NUM_REQS-1:0] greedy_index;
        wire [NUM_REQS-1:0]     greedy_onehot;

        VX_priority_encoder #(
            .N (NUM_REQS)
        ) greedy_sel (
            .data_in    (prev_grant),
            .index_out  (greedy_index),
            .onehot_out (greedy_onehot),
            `UNUSED_PIN (valid_out)
        );

        assign grant_valid  = |requests;
        assign grant_index  = retain_grant ? greedy_index  : oldest_index;
        assign grant_onehot = retain_grant ? greedy_onehot : oldest_onehot;

        `UNUSED_VAR (oldest_valid)

    end

endmodule
`TRACING_ON
