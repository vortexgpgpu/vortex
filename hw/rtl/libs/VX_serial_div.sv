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
module VX_serial_div #(
    parameter WIDTHN = 32,
    parameter WIDTHD = 32,
    parameter WIDTHQ = 32,
    parameter WIDTHR = 32,
    parameter LANES  = 1
) (
    input wire                      clk,
    input wire                      reset,

    input wire                      strobe,
    output wire                     busy,

    input wire                      is_signed,
    input wire [LANES-1:0][WIDTHN-1:0] numer,
    input wire [LANES-1:0][WIDTHD-1:0] denom,

    output wire [LANES-1:0][WIDTHQ-1:0] quotient,
    output wire [LANES-1:0][WIDTHR-1:0] remainder
);
    localparam MIN_ND = (WIDTHN < WIDTHD) ? WIDTHN : WIDTHD;
    localparam CNTRW = `CLOG2(WIDTHN);

    reg [LANES-1:0][WIDTHN + MIN_ND:0] working;
    reg [LANES-1:0][WIDTHD-1:0] denom_r;

    wire [LANES-1:0][WIDTHN-1:0] numer_qual;
    wire [LANES-1:0][WIDTHD-1:0] denom_qual;
    wire [LANES-1:0][WIDTHD:0] sub_result;

    reg [LANES-1:0] inv_quot, inv_rem;

    reg [CNTRW-1:0] cntr;
    reg busy_r;

    for (genvar i = 0; i < LANES; ++i) begin : g_setup
        wire negate_numer = is_signed && numer[i][WIDTHN-1];
        wire negate_denom = is_signed && denom[i][WIDTHD-1];
        assign numer_qual[i] = negate_numer ? -$signed(numer[i]) : numer[i];
        assign denom_qual[i] = negate_denom ? -$signed(denom[i]) : denom[i];
        assign sub_result[i] = working[i][WIDTHN + MIN_ND : WIDTHN] - denom_r[i];
    end

    always @(posedge clk) begin
        if (reset) begin
            busy_r <= 0;
        end else begin
            if (strobe) begin
                busy_r <= 1;
            end
            if (busy && cntr == 0) begin
                busy_r <= 0;
            end
        end
        cntr <= cntr - CNTRW'(1);
        if (strobe) begin
            cntr <= CNTRW'(WIDTHN-1);
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_div
        always @(posedge clk) begin
            if (strobe) begin
                working[i]  <= {{WIDTHD{1'b0}}, numer_qual[i], 1'b0};
                denom_r[i]  <= denom_qual[i];
                inv_quot[i] <= (denom[i] != 0) && is_signed && (numer[i][31] ^ denom[i][31]);
                inv_rem[i]  <= is_signed && numer[i][31];
            end else if (busy_r) begin
                working[i]  <= sub_result[i][WIDTHD] ? {working[i][WIDTHN+MIN_ND-1:0], 1'b0} :
                                                       {sub_result[i][WIDTHD-1:0], working[i][WIDTHN-1:0], 1'b1};
            end
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_output
        wire [WIDTHQ-1:0] q = working[i][WIDTHQ-1:0];
        wire [WIDTHR-1:0] r = working[i][WIDTHN+WIDTHR:WIDTHN+1];
        assign quotient[i]  = inv_quot[i] ? -$signed(q) : q;
        assign remainder[i] = inv_rem[i] ? -$signed(r) : r;
    end

    assign busy = busy_r;

endmodule
`TRACING_ON
