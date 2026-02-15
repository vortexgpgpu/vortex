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

module VX_tcu_tfr_pipe_register #(
    parameter SHARED_DATAW = 1,
    parameter LANE_DATAW   = 1,
    parameter NUM_LANES    = 1,
    parameter DEPTH        = 1,
    parameter PER_LANE_VALID = 0
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [NUM_LANES-1:0] lane_mask,

    // Shared Data (Scalar/Control)
    input wire [SHARED_DATAW-1:0] shared_data_in,
    output wire [SHARED_DATAW-1:0] shared_data_out,

    // Per-Lane Data (Vector) - Flattened
    input wire [NUM_LANES*LANE_DATAW-1:0] lane_data_in,
    output wire [NUM_LANES*LANE_DATAW-1:0] lane_data_out
);

    if (PER_LANE_VALID == 0) begin : g_merged
        VX_pipe_register #(
            .DATAW (SHARED_DATAW + (NUM_LANES * LANE_DATAW)),
            .DEPTH (DEPTH)
        ) pipe_merged (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({shared_data_in,  lane_data_in}),
            .data_out ({shared_data_out, lane_data_out})
        );
        `UNUSED_VAR (lane_mask)
    end else begin : g_split
        // 1. Shared Data Register
        VX_pipe_register #(
            .DATAW (SHARED_DATAW),
            .DEPTH (DEPTH)
        ) pipe_shared (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (shared_data_in),
            .data_out (shared_data_out)
        );
        // 2. Per-Lane Registers
        for (genvar i = 0; i < NUM_LANES; i++) begin : g_lanes
            VX_pipe_register #(
                .DATAW (LANE_DATAW),
                .DEPTH (DEPTH)
            ) pipe_lane (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable & lane_mask[i]),
                .data_in  (lane_data_in[i*LANE_DATAW +: LANE_DATAW]),
                .data_out (lane_data_out[i*LANE_DATAW +: LANE_DATAW])
            );
        end
    end

endmodule
