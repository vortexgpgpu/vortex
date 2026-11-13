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

// Clock gate for a fixed-latency pipeline that advances on `enable`: open
// while a request enters (valid_in, combinationally, so no wake-up cycle) or
// one entered in the last DEPTH enabled cycles, i.e. is still in flight, and
// during reset. ASIC only; elsewhere clk_out is clk_in.
`TRACING_OFF
module VX_pipe_clockgate #(
    parameter DEPTH = 1
) (
    input  wire clk_in,
    input  wire reset,
    input  wire enable,
    input  wire valid_in,
    output wire clk_out
);
    `STATIC_ASSERT (DEPTH > 0, ("invalid parameter"))

`ifdef ASIC
    localparam CNTW = `CLOG2(DEPTH + 1);

    reg [CNTW-1:0] in_flight;
    always @(posedge clk_in) begin
        if (reset) begin
            in_flight <= '0;
        end else if (enable) begin
            if (valid_in) begin
                in_flight <= CNTW'(DEPTH);
            end else if (in_flight != 0) begin
                in_flight <= in_flight - CNTW'(1);
            end
        end
    end

    VX_clockgate cg (
        .clk_in  (clk_in),
        .en      (reset || valid_in || (in_flight != 0)),
        .clk_out (clk_out)
    );
`else
    `UNUSED_VAR (reset)
    `UNUSED_VAR (enable)
    `UNUSED_VAR (valid_in)
    assign clk_out = clk_in;
`endif

endmodule
`TRACING_ON
