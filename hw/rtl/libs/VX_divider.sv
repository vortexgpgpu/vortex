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
module VX_divider #(
    parameter N_WIDTH  = 1,
    parameter D_WIDTH  = 1,
    parameter Q_WIDTH  = 1,
    parameter R_WIDTH  = 1,
    parameter N_SIGNED = 0,
    parameter D_SIGNED = 0,
    parameter LATENCY  = 0
) (
    input wire                clk,
    input wire                enable,
    input wire [N_WIDTH-1:0]  numer,
    input wire [D_WIDTH-1:0]  denom,
    output wire [Q_WIDTH-1:0] quotient,
    output wire [R_WIDTH-1:0] remainder
);

`ifdef QUARTUS

    wire [N_WIDTH-1:0] quotient_unqual;
    wire [D_WIDTH-1:0] remainder_unqual;

    lpm_divide divide (
        .clock    (clk),
        .clken    (enable),
        .numer    (numer),
        .denom    (denom),
        .quotient (quotient_unqual),
        .remain   (remainder_unqual)
    );

    defparam
        divide.lpm_type     = "LPM_DIVIDE",
        divide.lpm_widthn   = N_WIDTH,
        divide.lpm_widthd   = D_WIDTH,
        divide.lpm_nrepresentation = N_SIGNED ? "SIGNED" : "UNSIGNED",
        divide.lpm_drepresentation = D_SIGNED ? "SIGNED" : "UNSIGNED",
        divide.lpm_hint     = "MAXIMIZE_SPEED=6,LPM_REMAINDERPOSITIVE=FALSE",
        divide.lpm_pipeline = LATENCY;

    assign quotient  = quotient_unqual [Q_WIDTH-1:0];
    assign remainder = remainder_unqual [R_WIDTH-1:0];

`else

    reg [N_WIDTH-1:0] quotient_unqual;
    reg [D_WIDTH-1:0] remainder_unqual;

    always @(*) begin
        begin
            if (N_SIGNED && D_SIGNED) begin
                quotient_unqual  = $signed(numer) / $signed(denom);
                remainder_unqual = $signed(numer) % $signed(denom);
            end
            else if (N_SIGNED && !D_SIGNED) begin
                quotient_unqual  = $signed(numer) / denom;
                remainder_unqual = $signed(numer) % denom;
            end
            else if (!N_SIGNED && D_SIGNED) begin
                quotient_unqual  = numer / $signed(denom);
                remainder_unqual = numer % $signed(denom);
            end
            else begin
                quotient_unqual  = numer / denom;
                remainder_unqual = numer % denom;
            end
        end
    end

    if (LATENCY == 0) begin : g_comb
        assign quotient  = quotient_unqual [Q_WIDTH-1:0];
        assign remainder = remainder_unqual [R_WIDTH-1:0];
    end else begin : g_pipe
        reg [N_WIDTH-1:0] quotient_pipe [LATENCY-1:0];
        reg [D_WIDTH-1:0] remainder_pipe [LATENCY-1:0];

        for (genvar i = 0; i < LATENCY; ++i) begin : g_reg
            always @(posedge clk) begin
                if (enable) begin
                    quotient_pipe[i]  <= (0 == i) ? quotient_unqual  : quotient_pipe[i-1];
                    remainder_pipe[i] <= (0 == i) ? remainder_unqual : remainder_pipe[i-1];
                end
            end
        end

        assign quotient  = quotient_pipe[LATENCY-1][Q_WIDTH-1:0];
        assign remainder = remainder_pipe[LATENCY-1][R_WIDTH-1:0];
    end

`endif

endmodule
`TRACING_ON
