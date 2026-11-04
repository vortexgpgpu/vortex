// Copyright © 2019-2026
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

// The iterative multiplier and divider as VX_alu_muldiv builds them at
// XLEN 64: a 65 x 65 signed multiply and a 64-bit divide, two lanes each.
module VX_serial_muldiv_top (
    input wire              clk,
    input wire              reset,

    input wire              mul_strobe,
    output wire             mul_busy,
    input wire [1:0][64:0]  mul_a,
    input wire [1:0][64:0]  mul_b,
    output wire [1:0][129:0] mul_r,

    input wire              div_strobe,
    output wire             div_busy,
    input wire              div_signed,
    input wire [1:0][63:0]  div_n,
    input wire [1:0][63:0]  div_d,
    output wire [1:0][63:0] div_q,
    output wire [1:0][63:0] div_r
);
    VX_serial_mul #(
        .A_WIDTH (65),
        .LANES   (2),
        .SIGNED  (1)
    ) serial_mul (
        .clk    (clk),
        .reset  (reset),
        .strobe (mul_strobe),
        .busy   (mul_busy),
        .dataa  (mul_a),
        .datab  (mul_b),
        .result (mul_r)
    );

    VX_serial_div #(
        .WIDTHN (64),
        .WIDTHD (64),
        .WIDTHQ (64),
        .WIDTHR (64),
        .LANES  (2)
    ) serial_div (
        .clk       (clk),
        .reset     (reset),
        .strobe    (div_strobe),
        .busy      (div_busy),
        .is_signed (div_signed),
        .numer     (div_n),
        .denom     (div_d),
        .quotient  (div_q),
        .remainder (div_r)
    );
endmodule
