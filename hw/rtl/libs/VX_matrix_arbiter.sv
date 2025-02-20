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
module VX_matrix_arbiter #(
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
    if (NUM_REQS == 1)  begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (grant_ready)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin : g_arbiter

        reg [NUM_REQS-1:1] state [NUM_REQS-1:0];
        wire [NUM_REQS-1:0] pri [NUM_REQS-1:0];
        wire [NUM_REQS-1:0] grant;

        for (genvar r = 0; r < NUM_REQS; ++r) begin : g_pri_r
            for (genvar c = 0; c < NUM_REQS; ++c) begin : g_pri_c
                if (r > c) begin : g_row
                    assign pri[r][c] = requests[c] && state[c][r];
                end else if (r < c) begin : g_col
                    assign pri[r][c] = requests[c] && !state[r][c];
                end else begin : g_equal
                    assign pri[r][c] = 0;
                end
            end
        end

        for (genvar r = 0; r < NUM_REQS; ++r) begin : g_grant
            assign grant[r] = requests[r] && ~(| pri[r]);
        end

        for (genvar r = 0; r < NUM_REQS; ++r) begin : g_state_r
            for (genvar c = r + 1; c < NUM_REQS; ++c) begin : g_state_c
                always @(posedge clk) begin
                    if (reset) begin
                        state[r][c] <= '0;
                    end else if (grant_ready) begin
                        state[r][c] <= (state[r][c] || grant[c]) && ~grant[r];
                    end
                end
            end
        end

        assign grant_onehot = grant;

        VX_onehot_encoder #(
            .N (NUM_REQS)
        ) encoder (
            .data_in   (grant_onehot),
            .data_out  (grant_index),
            .valid_out (grant_valid)
        );
    end

endmodule
`TRACING_ON
