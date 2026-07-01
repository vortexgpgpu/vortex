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

module VX_tcu_tet_pipe_register #(
    parameter SHARED_DATAW = 1,
    parameter LANE_DATAW   = 1,
    parameter NUM_LANES    = 1,
    parameter DEPTH        = 1,
    parameter LANE_MASK    = 0
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [NUM_LANES-1:0] lane_mask,

    input wire [SHARED_DATAW-1:0] shared_data_in,
    output wire [SHARED_DATAW-1:0] shared_data_out,

    input wire [NUM_LANES*LANE_DATAW-1:0] lane_data_in,
    output wire [NUM_LANES*LANE_DATAW-1:0] lane_data_out
);

    if (LANE_MASK == 0) begin : g_merged
        (* shreg_extract = "no" *) reg [DEPTH-1:0][SHARED_DATAW + (NUM_LANES * LANE_DATAW)-1:0] pipe;

        always_ff @(posedge clk) begin
            if (reset) begin
                pipe <= '0;
            end else if (enable) begin
                pipe[0] <= {shared_data_in, lane_data_in};
                for (int i = 1; i < DEPTH; ++i) begin
                    pipe[i] <= pipe[i-1];
                end
            end
        end

        assign {shared_data_out, lane_data_out} = pipe[DEPTH-1];
        `UNUSED_VAR (lane_mask)
    end else begin : g_split
        (* shreg_extract = "no" *) reg [DEPTH-1:0][SHARED_DATAW-1:0] pipe_shared;
        (* shreg_extract = "no" *) reg [NUM_LANES-1:0][DEPTH-1:0][LANE_DATAW-1:0] pipe_lane;

        always_ff @(posedge clk) begin
            if (reset) begin
                pipe_shared <= '0;
            end else if (enable) begin
                pipe_shared[0] <= shared_data_in;
                for (int i = 1; i < DEPTH; ++i) begin
                    pipe_shared[i] <= pipe_shared[i-1];
                end
            end
        end

        for (genvar lane = 0; lane < NUM_LANES; ++lane) begin : g_lanes
            always_ff @(posedge clk) begin
                if (reset) begin
                    pipe_lane[lane] <= '0;
                end else if (enable) begin
                    pipe_lane[lane][0] <= lane_data_in[lane*LANE_DATAW +: LANE_DATAW];
                    for (int i = 1; i < DEPTH; ++i) begin
                        pipe_lane[lane][i] <= pipe_lane[lane][i-1];
                    end
                end
            end

            assign lane_data_out[lane*LANE_DATAW +: LANE_DATAW] = pipe_lane[lane][DEPTH-1];
        end

        assign shared_data_out = pipe_shared[DEPTH-1];
        `UNUSED_VAR (lane_mask)
    end

endmodule
