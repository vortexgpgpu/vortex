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
    localparam MIN_ND  = (WIDTHN < WIDTHD) ? WIDTHN : WIDTHD;
    localparam CNTRW   = `CLOG2(WIDTHN+2);
    localparam SUBW    = WIDTHD + 1;      // a step's trial subtract, with its borrow
    localparam L_WIDTH = SUBW / 2;        // its carry-select split

    // -x, carry-select at half width past 32 bits: -x = {-x_hi - b, -x_lo},
    // the borrow b being whether x_lo is nonzero (and -x_hi - 1 = ~x_hi)
    `define SERIAL_DIV_NEG(y, x, W) \
        if ((W) > 32) begin \
            assign y = {((| x[(W)/2-1:0]) ? ~x[(W)-1:(W)/2] : -x[(W)-1:(W)/2]), -x[(W)/2-1:0]}; \
        end else begin \
            assign y = -x; \
        end

    reg [LANES-1:0][WIDTHN + MIN_ND:0] working;
    reg [LANES-1:0][WIDTHD-1:0] denom_r;
    reg [LANES-1:0] neg_numer, neg_denom;

    // A first step takes the operands' magnitudes, WIDTHN division steps
    // follow, then a last step applies the result signs.
    reg [CNTRW-1:0] cntr;
    reg busy_r;
    reg abs_r;  // the step in progress takes the magnitudes
    reg fix_r;  // the step in progress applies the signs

    always @(posedge clk) begin
        if (reset) begin
            busy_r <= 0;
        end else begin
            if (strobe) begin
                busy_r <= 1;
            end
            if (busy_r && fix_r) begin
                busy_r <= 0;
            end
        end
        cntr  <= cntr - CNTRW'(1);
        abs_r <= 0;
        fix_r <= (cntr == CNTRW'(1));
        if (strobe) begin
            cntr  <= CNTRW'(WIDTHN+1);
            abs_r <= 1;
            fix_r <= 0;
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_div
        // trial subtract, carry-select at L_WIDTH: the upper part both ways
        // beside the lower part's borrow (a - b - 1 = a + ~b, one adder)
        wire [SUBW-1:0] rem_w = SUBW'(working[i][WIDTHN + MIN_ND : WIDTHN]);
        wire [SUBW-1:0] den_w = SUBW'(denom_r[i]);
        wire [L_WIDTH:0] lo = {1'b0, rem_w[L_WIDTH-1:0]} - {1'b0, den_w[L_WIDTH-1:0]};
        wire [SUBW-L_WIDTH-1:0] hi0 = rem_w[SUBW-1:L_WIDTH] - den_w[SUBW-1:L_WIDTH];
        wire [SUBW-L_WIDTH-1:0] hi1 = rem_w[SUBW-1:L_WIDTH] + ~den_w[SUBW-1:L_WIDTH];
        wire [SUBW-1:0] sub_result = {(lo[L_WIDTH] ? hi1 : hi0), lo[L_WIDTH-1:0]};

        wire [WIDTHN-1:0] n = working[i][WIDTHN:1];
        wire [WIDTHD-1:0] d = denom_r[i];
        wire [WIDTHQ-1:0] q = working[i][WIDTHQ-1:0];
        wire [WIDTHR-1:0] r = working[i][WIDTHN+WIDTHR:WIDTHN+1];
        wire [WIDTHN-1:0] n_neg;
        wire [WIDTHD-1:0] d_neg;
        wire [WIDTHQ-1:0] q_neg;
        wire [WIDTHR-1:0] r_neg;
        if (1) begin : g_n `SERIAL_DIV_NEG(n_neg, n, WIDTHN) end
        if (1) begin : g_d `SERIAL_DIV_NEG(d_neg, d, WIDTHD) end
        if (1) begin : g_q `SERIAL_DIV_NEG(q_neg, q, WIDTHQ) end
        if (1) begin : g_r `SERIAL_DIV_NEG(r_neg, r, WIDTHR) end

        // the quotient is negative for operands of opposite signs, but a
        // division by zero (all ones) is not; the remainder takes the
        // dividend's sign
        wire inv_quot = (neg_numer[i] ^ neg_denom[i]) && (d != 0);
        wire inv_rem  = neg_numer[i];

        always @(posedge clk) begin
            if (strobe) begin
                working[i]   <= {{WIDTHD{1'b0}}, numer[i], 1'b0};
                denom_r[i]   <= denom[i];
                neg_numer[i] <= is_signed && numer[i][WIDTHN-1];
                neg_denom[i] <= is_signed && denom[i][WIDTHD-1];
            end else if (busy_r) begin
                if (abs_r) begin
                    if (neg_numer[i]) begin
                        working[i][WIDTHN:1] <= n_neg;
                    end
                    if (neg_denom[i]) begin
                        denom_r[i] <= d_neg;
                    end
                end else if (fix_r) begin
                    if (inv_quot) begin
                        working[i][WIDTHQ-1:0] <= q_neg;
                    end
                    if (inv_rem) begin
                        working[i][WIDTHN+WIDTHR:WIDTHN+1] <= r_neg;
                    end
                end else begin
                    working[i] <= sub_result[WIDTHD] ? {working[i][WIDTHN+MIN_ND-1:0], 1'b0} :
                                                       {sub_result[WIDTHD-1:0], working[i][WIDTHN-1:0], 1'b1};
                end
            end
        end

        assign quotient[i]  = q;
        assign remainder[i] = r;
    end

    `undef SERIAL_DIV_NEG

    assign busy = busy_r;

endmodule

`TRACING_ON
