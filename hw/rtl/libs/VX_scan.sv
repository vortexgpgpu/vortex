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

// Fast Paralllel scan using Kogge-Stone style prefix tree with configurable operator
// Adapted from BaseJump STL: http://bjump.org/index.html

`TRACING_OFF
module VX_scan #(
    parameter N       = 1,
    parameter `STRING OP = "^",  // ^: XOR, &: AND, |: OR
    parameter REVERSE = 0        // 0: LO->HI, 1: HI->LO
) (
    input  wire [N-1:0] data_in,
    output wire [N-1:0] data_out
);
    localparam LOGN = `CLOG2(N);

`IGNORE_UNOPTFLAT_BEGIN
    wire [LOGN:0][N-1:0] t;
`IGNORE_UNOPTFLAT_END

    // reverses bits
    if (REVERSE != 0) begin
        assign t[0] = data_in;
    end else begin
        assign t[0] = {<<{data_in}};
    end

    // optimize for the common case of small and-scans
    if ((N == 2) && (OP == "&")) begin
	    assign t[LOGN] = {t[0][1], &t[0][1:0]};
    end else if ((N == 3) && (OP == "&")) begin
	    assign t[LOGN] = {t[0][2], &t[0][2:1], &t[0][2:0]};
    end else if ((N == 4) && (OP == "&")) begin
	    assign t[LOGN] = {t[0][3], &t[0][3:2], &t[0][3:1], &t[0][3:0]};
    end else begin
        // general case
        wire [N-1:0] fill;
	    for (genvar i = 0; i < LOGN; ++i) begin
            wire [N-1:0] shifted = N'({fill, t[i]} >> (1<<i));
            if (OP == "^") begin
		        assign fill = {N{1'b0}};
		        assign t[i+1] = t[i] ^ shifted;
            end else if (OP == "&") begin
		        assign fill = {N{1'b1}};
		        assign t[i+1] = t[i] & shifted;
            end else if (OP == "|") begin
		        assign fill = {N{1'b0}};
		        assign t[i+1] = t[i] | shifted;
            end
	    end
    end

    // reverse bits
    if (REVERSE != 0) begin
        assign data_out = t[LOGN];
    end else begin
        for (genvar i = 0; i < N; ++i) begin
            assign data_out[i] = t[LOGN][N-1-i];
        end
    end

endmodule
`TRACING_ON
