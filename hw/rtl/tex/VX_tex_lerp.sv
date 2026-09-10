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
    // Denominator the weight is normalized against. A bilinear tap weight is a
    // fraction of 255, matching the packed-colour blend the software sampler
    // uses for the same taps; a mip-level weight is a fraction of 256. The two
    // round differently by up to one count, so a sample filtered on this unit
    // only reproduces the software sampler if each weight keeps its own form.
    parameter FRAC_SCALE = 255
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
    `STATIC_ASSERT(FRAC_SCALE == 255 || FRAC_SCALE == 256, ("invalid value"))

    if (FRAC_SCALE == 256) begin : g_scale_256
        reg [15:0] p1, p2;
        reg [15:0] sum;
        reg [7:0]  res;
        // The blend is a fraction of 256, so the result is the high byte and the
        // low one is the remainder this form truncates rather than rounds.
        `UNUSED_VAR (sum[7:0])

        wire [8:0] sub = (9'h100 - 9'(frac));

        always @(posedge clk) begin
            if (enable) begin
                p1  <= 16'(in1 * sub);
                p2  <= 16'(in2 * frac);
                sum <= p1 + p2;
                res <= sum[15:8];
            end
        end

        assign out = res;
    end else begin : g_scale_255
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
    end

endmodule
