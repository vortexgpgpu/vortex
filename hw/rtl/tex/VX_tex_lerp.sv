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

module VX_tex_lerp #(
    parameter LATENCY = 3,
    // The weight is frac/256 (8 subtexel fraction bits). A bilinear tap blend
    // rounds to nearest; the mip-level blend truncates, as the software
    // sampler's level blend does, so both paths keep producing the same texel.
    parameter ROUND = 1
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
    `STATIC_ASSERT(ROUND == 0 || ROUND == 1, ("invalid value"))

    localparam [15:0] BIAS = ROUND ? 16'h80 : 16'h0;

    reg [15:0] p1, p2;
    reg [15:0] sum;
    reg [7:0]  res;
    // The result is the high byte; the low one is the discarded remainder.
    `UNUSED_VAR (sum[7:0])

    wire [8:0] sub = (9'h100 - 9'(frac));

    // 255*256 + 128 < 2^16: the 16-bit accumulator cannot overflow.
    always @(posedge clk) begin
        if (enable) begin
            p1  <= 16'(in1 * sub);
            p2  <= 16'(in2 * frac);
            sum <= p1 + p2 + BIAS;
            res <= sum[15:8];
        end
    end

    assign out = res;

endmodule
