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
`include "HardFloat_consts.vi"

module VX_tcu_bhf_fp8mul #(
    parameter IN_EXPW  = 4,
    parameter IN_SIGW  = 3,     // Includes implicit bit
    parameter OUT_EXPW = IN_EXPW,
    parameter OUT_SIGW = IN_SIGW, // Includes implicit bit
    parameter MUL_LATENCY = 1,
    parameter RND_LATENCY = 1,
    parameter IN_REC   = 0,     // 0: IEEE754, 1: recoded
    parameter OUT_REC  = 0,     // 0: IEEE754, 1: recoded
    parameter IN_FECW = IN_EXPW + IN_SIGW + IN_REC,
    parameter OUT_FECW = OUT_EXPW + OUT_SIGW + OUT_REC
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [2:0]       frm,
    input  wire [IN_FECW-1:0] a0,
    input  wire [IN_FECW-1:0] b0,
    input  wire [IN_FECW-1:0] a1,
    input  wire [IN_FECW-1:0] b1,
    output wire [OUT_FECW-1:0] y,
    output wire [4:0]      fflags
);
    localparam MUL_EXPW = IN_EXPW + 1;
    localparam MUL_SIGW = 2 * (IN_EXPW + IN_SIGW) + 7;
    localparam MUL_RECW = MUL_EXPW + MUL_SIGW + 1;

    wire [MUL_RECW-1:0] y0, y1;

    VX_tcu_bhf_fmul #(
        .IN_EXPW (IN_EXPW),
        .IN_SIGW (IN_SIGW),
        .OUT_EXPW(MUL_EXPW),
        .OUT_SIGW(MUL_SIGW),
        .IN_REC  (IN_REC),  // input in IEEE format
        .OUT_REC (1),       // output in recoded format
        .MUL_LATENCY (0),
        .RND_LATENCY (MUL_LATENCY)
    ) fp8_mul_0 (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .frm    (frm),
        .a      (a0),
        .b      (b0),
        .y      (y0),
        `UNUSED_PIN(fflags)
    );

    VX_tcu_bhf_fmul #(
        .IN_EXPW (IN_EXPW),
        .IN_SIGW (IN_SIGW),
        .OUT_EXPW(MUL_EXPW),
        .OUT_SIGW(MUL_SIGW),
        .IN_REC  (IN_REC),  // input in IEEE format
        .OUT_REC (1),       // output in recoded format
        .MUL_LATENCY (0),
        .RND_LATENCY (MUL_LATENCY)
    ) fp8_mul_1 (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .frm    (frm),
        .a      (a1),
        .b      (b1),
        .y      (y1),
        `UNUSED_PIN(fflags)
    );

    VX_tcu_bhf_fadd #(
        .IN_EXPW (MUL_EXPW),
        .IN_SIGW (MUL_SIGW),
        .OUT_EXPW(OUT_EXPW),
        .OUT_SIGW(OUT_SIGW),
        .IN_REC  (1),       // input in recoded format
        .OUT_REC (OUT_REC), // output in IEEE format
        .ADD_LATENCY (0),
        .RND_LATENCY (RND_LATENCY)
    ) fp16_add (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .frm    (frm),
        .a      (y0),
        .b      (y1),
        .y      (y),
        .fflags (fflags)
    );

endmodule
