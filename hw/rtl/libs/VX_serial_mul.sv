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

    localparam CNTRW = `CLOG2(X_WIDTH);

    reg [LANES-1:0][X_WIDTH-1:0] a;
	reg [LANES-1:0][Y_WIDTH-1:0] b;
	reg [LANES-1:0][P_WIDTH-1:0] p;

    reg [CNTRW-1:0] cntr;
    reg busy_r;

    always @(posedge clk) begin
        if (reset) begin
            busy_r <= 0;
        end else begin
            if (strobe) begin
                busy_r <= 1;
            end
            if (busy_r && cntr == 0) begin
                busy_r <= 0;
            end
        end
        cntr <= cntr - CNTRW'(1);
        if (strobe) begin
            cntr <= CNTRW'(X_WIDTH-1);
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_mul
        wire [X_WIDTH-1:0] axb = b[i][0] ? a[i] : '0;

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
                if (SIGNED) begin
                    if (cntr == 0) begin
                        p[i][P_WIDTH-1:Y_WIDTH-1] <= {1'b0, p[i][P_WIDTH-1:Y_WIDTH]} + {1'b0, axb[X_WIDTH-1], ~axb[X_WIDTH-2:0]};
                    end else begin
                        p[i][P_WIDTH-1:Y_WIDTH-1] <= {1'b0, p[i][P_WIDTH-1:Y_WIDTH]} + {1'b0, ~axb[X_WIDTH-1], axb[X_WIDTH-2:0]};
                    end
                end else begin
                    p[i][P_WIDTH-1:Y_WIDTH-1] <= {1'b0, p[i][P_WIDTH-1:Y_WIDTH]} + ((b[i][0]) ? {1'b0, a[i]} : 0);
                end
            end
        end

        if (SIGNED) begin : g_signed
            assign result[i] = R_WIDTH'(p[i][P_WIDTH-1:0] + {1'b1, {(X_WIDTH-2){1'b0}}, 1'b1, {(Y_WIDTH){1'b0}}});
        end else begin : g_unsigned
            assign result[i] = R_WIDTH'(p[i]);
        end
    end
    `UNUSED_VAR (p)

    assign busy = busy_r;

endmodule
`TRACING_ON
