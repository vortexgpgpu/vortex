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

// One lane of VX_serial_div: a restoring divider retiring one quotient bit per
// cycle while busy, on operands made positive at strobe and sign-corrected at
// the output. The shared iteration counter stays in VX_serial_div.
`TRACING_OFF
module VX_serial_div_lane #(
    parameter WIDTHN = 32,
    parameter WIDTHD = 32,
    parameter WIDTHQ = 32,
    parameter WIDTHR = 32
) (
    input wire                  clk,

    input wire                  strobe,
    input wire                  busy,

    input wire                  is_signed,
    input wire [WIDTHN-1:0]     numer,
    input wire [WIDTHD-1:0]     denom,

    output wire [WIDTHQ-1:0]    quotient,
    output wire [WIDTHR-1:0]    remainder
);
    localparam MIN_ND = (WIDTHN < WIDTHD) ? WIDTHN : WIDTHD;

    reg [WIDTHN + MIN_ND:0] working;
    reg [WIDTHD-1:0] denom_r;
    reg inv_quot, inv_rem;

    wire negate_numer = is_signed && numer[WIDTHN-1];
    wire negate_denom = is_signed && denom[WIDTHD-1];
    wire [WIDTHN-1:0] numer_qual = negate_numer ? -$signed(numer) : numer;
    wire [WIDTHD-1:0] denom_qual = negate_denom ? -$signed(denom) : denom;
    wire [WIDTHD:0] sub_result = working[WIDTHN + MIN_ND : WIDTHN] - denom_r;

    always @(posedge clk) begin
        if (strobe) begin
            working  <= {{WIDTHD{1'b0}}, numer_qual, 1'b0};
            denom_r  <= denom_qual;
            inv_quot <= (denom != 0) && is_signed && (numer[31] ^ denom[31]);
            inv_rem  <= is_signed && numer[31];
        end else if (busy) begin
            working  <= sub_result[WIDTHD] ? {working[WIDTHN+MIN_ND-1:0], 1'b0} :
                                             {sub_result[WIDTHD-1:0], working[WIDTHN-1:0], 1'b1};
        end
    end

    wire [WIDTHQ-1:0] q = working[WIDTHQ-1:0];
    wire [WIDTHR-1:0] r = working[WIDTHN+WIDTHR:WIDTHN+1];
    assign quotient  = inv_quot ? -$signed(q) : q;
    assign remainder = inv_rem ? -$signed(r) : r;

endmodule
`TRACING_ON
