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

`TRACING_OFF
// Two-level least-recently-granted arbiter: requesters form groups of GROUP;
// the least recently granted group with a request wins, then its least
// recently granted member. A full matrix arbiter's state grows with the square
// of its requesters, this one's with GROUP times them.
module VX_matrix2_arbiter #(
    parameter NUM_REQS     = 1,
    parameter GROUP        = 8,
    parameter STICKY       = 0, // hold the grant until its request is deasserted
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
    localparam NUM_GROUPS = `CDIV(NUM_REQS, GROUP);
    localparam PADDED     = NUM_GROUPS * GROUP;
    localparam NG_BITS    = `LOG2UP(NUM_GROUPS);
    localparam GM_BITS    = `LOG2UP(GROUP);

    `STATIC_ASSERT ((GROUP & (GROUP - 1)) == 0, ("GROUP must be a power of two"))

    wire [PADDED-1:0] reqs_p = PADDED'(requests);

    reg [NUM_REQS-1:0] prev_grant;
    wire retain_grant = (STICKY != 0) && (|(prev_grant & requests));
    // the orders move only on a new grant, as in VX_matrix_arbiter
    wire order_ready = grant_ready && ~retain_grant;

    wire [NUM_GROUPS-1:0] group_reqs;
    for (genvar g = 0; g < NUM_GROUPS; ++g) begin : g_group_reqs
        assign group_reqs[g] = | reqs_p[g * GROUP +: GROUP];
    end

    wire [NG_BITS-1:0]    group_index;
    wire [NUM_GROUPS-1:0] group_onehot;
    wire                  group_valid;

    VX_matrix_arbiter #(
        .NUM_REQS (NUM_GROUPS)
    ) group_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (group_reqs),
        .grant_index  (group_index),
        .grant_onehot (group_onehot),
        .grant_valid  (group_valid),
        .grant_ready  (order_ready)
    );
    `UNUSED_VAR ({group_index, group_valid})

    wire [PADDED-1:0] grant_p;
    for (genvar g = 0; g < NUM_GROUPS; ++g) begin : g_members
        wire [GM_BITS-1:0] member_index;
        wire [GROUP-1:0]   member_onehot;
        wire               member_valid;
        VX_matrix_arbiter #(
            .NUM_REQS (GROUP)
        ) member_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (reqs_p[g * GROUP +: GROUP]),
            .grant_index  (member_index),
            .grant_onehot (member_onehot),
            .grant_valid  (member_valid),
            .grant_ready  (order_ready && group_onehot[g])
        );
        `UNUSED_VAR ({member_index, member_valid})
        assign grant_p[g * GROUP +: GROUP] = member_onehot & {GROUP{group_onehot[g]}};
    end
    if (PADDED > NUM_REQS) begin : g_pad
        `UNUSED_VAR (grant_p[PADDED-1:NUM_REQS])
    end

    wire [NUM_REQS-1:0] grant_w = retain_grant ? prev_grant : grant_p[NUM_REQS-1:0];

    always @(posedge clk) begin
        if (reset) begin
            prev_grant <= '0;
        end else if (grant_valid && grant_ready) begin
            prev_grant <= grant_w;
        end
    end

    VX_onehot_encoder #(
        .N (NUM_REQS)
    ) encoder (
        .data_in   (grant_w),
        .data_out  (grant_index),
        `UNUSED_PIN (valid_out)
    );

    assign grant_onehot = grant_w;
    assign grant_valid  = (| requests);

endmodule
`TRACING_ON
