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
module VX_shift_register #(
    parameter DATAW      = 1,
    parameter RESETW     = 0,
    parameter DEPTH      = 1,
    parameter NUM_TAPS   = 1,
    parameter TAP_START  = 0,
    parameter TAP_STRIDE = 1
) (
    input wire                         clk,
    input wire                         reset,
    input wire                         enable,
    input wire [DATAW-1:0]             data_in,
    output wire [NUM_TAPS-1:0][DATAW-1:0] data_out
);
    if (DEPTH != 0) begin : g_shift_register
        reg [DEPTH-1:0][DATAW-1:0] entries;

        always @(posedge clk) begin
            for (integer i = 0; i < DATAW; ++i) begin
                if ((i >= (DATAW-RESETW)) && reset) begin
                    for (integer j = 0; j < DEPTH; ++j)
                        entries[j][i] <= 0;
                end else if (enable) begin
                    for (integer j = 1; j < DEPTH; ++j)
                        entries[j-1][i] <= entries[j][i];
                    entries[DEPTH-1][i] <= data_in[i];
                end
            end
        end

        for (genvar i = 0; i < NUM_TAPS; ++i) begin : g_data_out
            assign data_out[i] = entries[i * TAP_STRIDE + TAP_START];
        end
    end else begin : g_passthru
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (enable)
        assign data_out = data_in;
    end

endmodule
`TRACING_ON
