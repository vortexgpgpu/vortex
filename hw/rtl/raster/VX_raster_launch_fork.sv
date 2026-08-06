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

// Distribution stage for the delegated-launch kick (see
// VX_raster_launch_if): an eager VX_stream_fork of the single-token stream.
// Eager mode tracks per-engine acceptance (delivered_r), so the master's
// handshake completes only when every engine has consumed the kick — the
// KMU's busy therefore covers the kick until the last engine has taken over
// with its own frame busy, leaving the launch fence no gap to observe.
module VX_raster_launch_fork #(
    parameter NUM_OUTPUTS = 1
) (
    input wire clk,
    input wire reset,

    VX_raster_launch_if.slave  bus_in_if,
    VX_raster_launch_if.master bus_out_if [NUM_OUTPUTS]
);
    wire [NUM_OUTPUTS-1:0] valid_out;
    wire [NUM_OUTPUTS-1:0] ready_out;

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_bus_out
        assign bus_out_if[i].valid = valid_out[i];
        assign ready_out[i] = bus_out_if[i].ready;
    end

    wire [NUM_OUTPUTS-1:0][0:0] data_out;
    `UNUSED_VAR (data_out)

    VX_stream_fork #(
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (1),
        .EAGER       (1)
    ) stream_fork (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.valid),
        .data_in   (1'b0),
        .ready_in  (bus_in_if.ready),
        .valid_out (valid_out),
        .data_out  (data_out),
        .ready_out (ready_out)
    );

endmodule
