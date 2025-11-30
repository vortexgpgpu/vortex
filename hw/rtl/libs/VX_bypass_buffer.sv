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

// A bypass elastic buffer operates at full bandwidth where pop can happen if the buffer is empty but is going full
// It has the following benefits:
// + Full-bandwidth throughput
// + use only one register for storage
// It has the following limitations:
// + data_out is not registered
// + ready_in and ready_out are coupled

`include "VX_platform.vh"

`TRACING_OFF
module VX_bypass_buffer #(
    parameter DATAW    = 1,
    parameter PASSTHRU = 0
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
        assign ready_in  = ready_out;
        assign valid_out = valid_in;
        assign data_out  = data_in;

    end else begin : g_buffer

        reg [DATAW-1:0] buffer;
        reg has_data;

        always @(posedge clk) begin
            if (reset) begin
                has_data <= 0;
            end else begin
                if (ready_out) begin
                    has_data <= 0;
                end else if (~has_data) begin
                    has_data <= valid_in;
                end
            end
            if (~has_data) begin
                buffer <= data_in;
            end
        end

        assign ready_in  = ready_out || ~has_data;
        assign data_out  = has_data ? buffer : data_in;
        assign valid_out = valid_in || has_data;

    end

endmodule
`TRACING_ON
