//!/bin/bash

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
module VX_tex_lerp #(
    parameter LATENCY = 3
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [7:0]  in1,
    input wire [7:0]  in2,
    input wire [7:0]  frac,
    output wire [7:0] out
);
    `UNUSED_VAR (reset)
    `STATIC_ASSERT(LATENCY == 3, ("invalid value"))

    reg [15:0] p1, p2;
    reg [15:0] sum;
    reg [7:0]  res;

    wire [7:0] sub = (8'hff - frac);

    always @(posedge clk) begin
        if (enable) begin
            p1  <= in1 * sub;
            p2  <= in2 * frac;
            sum <= p1 + p2 + 16'h80;
            res <= 8'((sum + (sum >> 8)) >> 8);
        end
    end

    assign out = res;

endmodule
`TRACING_ON
