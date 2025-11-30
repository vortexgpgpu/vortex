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
module VX_skid_buffer #(
    parameter DATAW    = 32,
    parameter PASSTHRU = 0,
    parameter HALF_BW  = 0,
    parameter OUT_REG  = 0
) (
    input  wire             clk,
    input  wire             reset,

    input  wire             valid_in,
    output wire             ready_in,
    input  wire [DATAW-1:0] data_in,

    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
);
    if (PASSTHRU != 0) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end else if (HALF_BW != 0) begin : g_half_bw

        VX_toggle_buffer #(
            .DATAW (DATAW)
        ) toggle_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end else begin : g_full_bw

        VX_stream_buffer #(
            .DATAW (DATAW),
            .OUT_REG (OUT_REG)
        ) stream_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end

endmodule
`TRACING_ON
