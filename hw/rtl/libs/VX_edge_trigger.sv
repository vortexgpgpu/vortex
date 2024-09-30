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
module VX_edge_trigger #(
    parameter POS  = 0,
    parameter INIT = 0
) (
    input wire  clk,
    input wire  reset,
    input wire  data_in,
    output wire data_out
);
    reg prev;

    always @(posedge clk) begin
        if (reset) begin
            prev <= INIT;
        end else begin
            prev <= data_in;
        end
    end

    if (POS != 0) begin : g_pos
        assign data_out = data_in & ~prev;
    end else begin : g_neg
        assign data_out = ~data_in & prev;
    end

endmodule
`TRACING_ON
