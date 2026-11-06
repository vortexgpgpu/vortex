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

// Both VX_tex_lerp forms the sampler instantiates, on shared inputs.
module VX_tex_lerp_top (
    input wire        clk,
    input wire        reset,
    input wire        enable,
    input wire [7:0]  in1,
    input wire [7:0]  in2,
    input wire [7:0]  frac,
    output wire [7:0] out_round,
    output wire [7:0] out_trunc
);
    VX_tex_lerp #(
        .LATENCY (3),
        .ROUND   (1)
    ) lerp_round (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .in1    (in1),
        .in2    (in2),
        .frac   (frac),
        .out    (out_round)
    );

    VX_tex_lerp #(
        .LATENCY (3),
        .ROUND   (0)
    ) lerp_trunc (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .in1    (in1),
        .in2    (in2),
        .frac   (frac),
        .out    (out_trunc)
    );

endmodule
