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

// Iterative integer multiplier
// An adaptation of ZipCPU algorithm for a multi-lane elastic architecture.
// https://zipcpu.com/zipcpu/2021/07/03/slowmpy.html

`TRACING_OFF
module VX_serial_mul #(
    parameter A_WIDTH = 32,
    parameter B_WIDTH = A_WIDTH,
    parameter R_WIDTH = A_WIDTH + B_WIDTH,
    parameter SIGNED  = 0,
    parameter LANES   = 1
) (
    input wire                          clk,
    input wire                          reset,

    input wire                          strobe,
    output wire                         busy,

    input wire [LANES-1:0][A_WIDTH-1:0] dataa,
    input wire [LANES-1:0][B_WIDTH-1:0] datab,
    output wire [LANES-1:0][R_WIDTH-1:0] result
);
    localparam X_WIDTH = SIGNED ? `MAX(A_WIDTH, B_WIDTH) : A_WIDTH;
    localparam Y_WIDTH = SIGNED ? `MAX(A_WIDTH, B_WIDTH) : B_WIDTH;
    localparam P_WIDTH = X_WIDTH + Y_WIDTH;
    localparam S_WIDTH = X_WIDTH + 1;     // a step's partial sum
    localparam L_WIDTH = S_WIDTH / 2;     // its carry-select split

    localparam CNTRW = `CLOG2(X_WIDTH);

    reg [LANES-1:0][X_WIDTH-1:0] a;
    reg [LANES-1:0][Y_WIDTH-1:0] b;
    reg [LANES-1:0][P_WIDTH-1:0] p;

    reg [CNTRW-1:0] cntr;
    reg busy_r;
    reg last_r; // the step in progress is the last one (cntr == 0)

    always @(posedge clk) begin
        if (reset) begin
            busy_r <= 0;
        end else begin
            if (strobe) begin
                busy_r <= 1;
            end
            if (busy_r && last_r) begin
                busy_r <= 0;
            end
        end
        cntr   <= cntr - CNTRW'(1);
        last_r <= (cntr == CNTRW'(1));
        if (strobe) begin
            cntr   <= CNTRW'(X_WIDTH-1);
            last_r <= (X_WIDTH == 1);
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_mul
        // Shift-and-add over the multiplier's bits. Signed, the last bit has
        // a negative weight, so the last step subtracts: the multiplicand is
        // complemented and the carry-in is set, and the partial sum is one
        // bit wider than the product's upper half, so it never overflows.
        wire [S_WIDTH-1:0] p_hi = SIGNED ? {p[i][P_WIDTH-1], p[i][P_WIDTH-1:Y_WIDTH]}
                                         : {1'b0, p[i][P_WIDTH-1:Y_WIDTH]};
        wire [S_WIDTH-1:0] axb  = b[i][0] ? (SIGNED ? {a[i][X_WIDTH-1], a[i]} : {1'b0, a[i]}) : '0;
        wire               neg  = SIGNED && last_r;
        wire [S_WIDTH-1:0] addend = axb ^ {S_WIDTH{neg}};

        // carry-select at L_WIDTH: the upper part both ways beside the
        // lower part's carry, each a single adder (a carry-in rides in a
        // forced low bit)
        wire [L_WIDTH+1:0] lo = {1'b0, p_hi[L_WIDTH-1:0], 1'b1} + {1'b0, addend[L_WIDTH-1:0], neg};
        wire [S_WIDTH-L_WIDTH-1:0] hi0 = p_hi[S_WIDTH-1:L_WIDTH] + addend[S_WIDTH-1:L_WIDTH];
        wire [S_WIDTH-L_WIDTH:0]   hi1 = {p_hi[S_WIDTH-1:L_WIDTH], 1'b1} + {addend[S_WIDTH-1:L_WIDTH], 1'b1};
        wire [S_WIDTH-1:0] sum = {(lo[L_WIDTH+1] ? hi1[S_WIDTH-L_WIDTH:1] : hi0), lo[L_WIDTH:1]};
        `UNUSED_VAR (lo[0])
        `UNUSED_VAR (hi1[0])

        always @(posedge clk) begin
            if (strobe) begin
                if (SIGNED) begin
                    a[i] <= X_WIDTH'($signed(dataa[i]));
                    b[i] <= Y_WIDTH'($signed(datab[i]));
                end else begin
                    a[i] <= dataa[i];
                    b[i] <= datab[i];
                end
                p[i] <= 0;
            end else if (busy_r) begin
                b[i] <= (b[i] >> 1);
                p[i][Y_WIDTH-2:0] <= p[i][Y_WIDTH-1:1];
                p[i][P_WIDTH-1:Y_WIDTH-1] <= sum;
            end
        end

        assign result[i] = R_WIDTH'(p[i]);
    end

    assign busy = busy_r;

endmodule
`TRACING_ON
