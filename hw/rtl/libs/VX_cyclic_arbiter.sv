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
module VX_cyclic_arbiter #(
    parameter NUM_REQS     = 1,
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
    if (NUM_REQS == 1) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (grant_ready)
        `UNUSED_PARAM (STICKY)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin : g_arbiter

        localparam IS_POW2 = (1 << LOG_NUM_REQS) == NUM_REQS;

        wire [LOG_NUM_REQS-1:0] grant_index_um;
        wire [NUM_REQS-1:0] grant_onehot_w, grant_onehot_um;
        reg [LOG_NUM_REQS-1:0] grant_index_r;

        reg [NUM_REQS-1:0] prev_grant;

        always @(posedge clk) begin
            if (reset) begin
                prev_grant <= '0;
            end else if (grant_valid && grant_ready) begin
                prev_grant <= grant_onehot;
            end
        end

        wire retain_grant = (STICKY != 0) && (|(prev_grant & requests));

        wire [NUM_REQS-1:0] requests_w = retain_grant ? prev_grant : requests;

        always @(posedge clk) begin
            if (reset) begin
                grant_index_r <= '0;
            end else if (grant_valid && grant_ready && ~retain_grant) begin
                if (!IS_POW2 && grant_index == LOG_NUM_REQS'(NUM_REQS-1)) begin
                    grant_index_r <= '0;
                end else begin
                    grant_index_r <= grant_index + LOG_NUM_REQS'(1);
                end
            end
        end

        wire grant_valid_w;

        VX_priority_encoder #(
            .N (NUM_REQS)
        ) grant_sel (
            .data_in    (requests_w),
            .onehot_out (grant_onehot_um),
            .index_out  (grant_index_um),
            .valid_out  (grant_valid_w)
        );

        VX_demux #(
            .DATAW (1),
            .N (NUM_REQS)
        ) grant_decoder (
            .sel_in   (grant_index_r),
            .data_in  (1'b1),
            .data_out (grant_onehot_w)
        );

        wire is_hit = requests[grant_index_r] && ~retain_grant;

        assign grant_index  = is_hit ? grant_index_r : grant_index_um;
        assign grant_onehot = is_hit ? grant_onehot_w : grant_onehot_um;
        assign grant_valid  = (STICKY != 0) ? (| requests) : grant_valid_w;

    end

endmodule
`TRACING_ON
