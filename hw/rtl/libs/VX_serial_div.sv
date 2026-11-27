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
    localparam CNTRW = `CLOG2(WIDTHN);

    reg [CNTRW-1:0] cntr;
    reg busy_r;

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

    for (genvar i = 0; i < LANES; ++i) begin : g_lanes
        VX_serial_div_lane #(
            .WIDTHN (WIDTHN),
            .WIDTHD (WIDTHD),
            .WIDTHQ (WIDTHQ),
            .WIDTHR (WIDTHR)
        ) div_lane (
            .clk       (clk),
            .strobe    (strobe),
            .busy      (busy_r),
            .is_signed (is_signed),
            .numer     (numer[i]),
            .denom     (denom[i]),
            .quotient  (quotient[i]),
            .remainder (remainder[i])
        );
    end

    assign busy = busy_r;

endmodule
`TRACING_ON
