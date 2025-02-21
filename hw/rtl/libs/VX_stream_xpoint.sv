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
module VX_stream_xpoint #(
    parameter NUM_INPUTS    = 1,
    parameter NUM_OUTPUTS   = 1,
    parameter DATAW         = 1,
    parameter OUT_DRIVEN    = 0,
    parameter OUT_BUF       = 0,
    parameter SEL_SRC       = OUT_DRIVEN ? NUM_OUTPUTS : NUM_INPUTS,
    parameter SEL_DST       = OUT_DRIVEN ? NUM_INPUTS : NUM_OUTPUTS
) (
    input wire                              clk,
    input wire                              reset,

    input wire  [SEL_SRC-1:0][`LOG2UP(SEL_DST)-1:0] sel_in,

    input  wire [NUM_INPUTS-1:0]            valid_in,
    input  wire [NUM_INPUTS-1:0][DATAW-1:0] data_in,
    output wire [NUM_INPUTS-1:0]            ready_in,

    output wire [NUM_OUTPUTS-1:0]           valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out,
    input  wire [NUM_OUTPUTS-1:0]           ready_out
);
    logic [NUM_OUTPUTS-1:0]            valid_out_w;
    logic [NUM_OUTPUTS-1:0][DATAW-1:0] data_out_w;
    logic [NUM_OUTPUTS-1:0]            ready_out_w;

    if (OUT_DRIVEN) begin : g_output_driven

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf
            assign valid_out_w[o] = valid_in[sel_in[o]];
            assign data_out_w[o] = data_in[sel_in[o]];
        end

        logic [NUM_INPUTS-1:0] ready_in_w;
        always @(*) begin
            ready_in_w = '0;
            for (integer o = 0; o < NUM_OUTPUTS; ++o) begin
                ready_in_w[sel_in[o]] = ready_out_w[o];
            end
        end
        assign ready_in = ready_in_w;

    end else begin: g_input_driven

        always @(*) begin
            valid_out_w = '0;
            data_out_w = 'x;
            for (integer i = 0; i < NUM_INPUTS; ++i) begin
                if (valid_in[i]) begin
                    valid_out_w[sel_in[i]] = 1;
                    data_out_w[sel_in[i]] = data_in[i];
                end
            end
        end

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_ready_in
            assign ready_in[i] = ready_out_w[sel_in[i]];
        end
    end

    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf
        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_out_w[o]),
            .data_in   (data_out_w[o]),
            .ready_in  (ready_out_w[o]),
            .valid_out (valid_out[o]),
            .data_out  (data_out[o]),
            .ready_out (ready_out[o])
        );
    end

endmodule
`TRACING_ON
