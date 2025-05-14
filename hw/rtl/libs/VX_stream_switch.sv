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
module VX_stream_switch #(
    parameter NUM_INPUTS    = 1,
    parameter NUM_OUTPUTS   = 1,
    parameter DATAW         = 1,
    parameter OUT_BUF       = 0,
    parameter NUM_REQS      = (NUM_INPUTS > NUM_OUTPUTS) ? `CDIV(NUM_INPUTS, NUM_OUTPUTS) : `CDIV(NUM_OUTPUTS, NUM_INPUTS),
    parameter SEL_COUNT     = `MIN(NUM_INPUTS, NUM_OUTPUTS),
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS)
) (
    input wire                              clk,
    input wire                              reset,

    input wire  [SEL_COUNT-1:0][`UP(LOG_NUM_REQS)-1:0] sel_in,

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

    if (NUM_INPUTS > NUM_OUTPUTS) begin : g_input_select

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf

            logic [NUM_REQS-1:0] valid_in_w;
            logic [NUM_REQS-1:0][DATAW-1:0] data_in_w;
            logic [NUM_REQS-1:0] ready_in_w;

            for (genvar r = 0; r < NUM_REQS; ++r) begin : g_r
                localparam i = r * NUM_OUTPUTS + o;
                if (i < NUM_INPUTS) begin : g_valid
                    assign valid_in_w[r] = valid_in[i];
                    assign data_in_w[r]  = data_in[i];
                    assign ready_in[i]   = ready_in_w[r];
                end else begin : g_padding
                    assign valid_in_w[r] = 0;
                    assign data_in_w[r]  = '0;
                    `UNUSED_VAR (ready_in_w[r])
                end
            end

            assign valid_out_w[o] = valid_in_w[sel_in[o]];
            assign data_out_w[o] = data_in_w[sel_in[o]];

            always @(*) begin
                ready_in_w = '0;
                for (integer o = 0; o < NUM_OUTPUTS; ++o) begin
                    ready_in_w[sel_in[o]] = ready_out_w[o];
                end
            end
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin : g_output_select

        // Inputs < Outputs

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_out_buf

            logic [NUM_REQS-1:0] ready_out_s;

            for (genvar r = 0; r < NUM_REQS; ++r) begin : g_r
                localparam o = r * NUM_INPUTS + i;
                if (o < NUM_OUTPUTS) begin : g_valid
                    assign valid_out_w[o] = valid_in[i] && (sel_in[i] == LOG_NUM_REQS'(r));
                    assign data_out_w[o] = data_in[i];
                    assign ready_out_s[r] = ready_out_w[o];
                end else begin : g_padding
                    assign ready_out_s[r] = '0;
                end
            end

            assign ready_in[i] = ready_out_s[sel_in[i]];
        end

    end else begin : g_passthru

        // #Inputs == #Outputs

        `UNUSED_VAR (sel_in)

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
            assign valid_out_w[i] = valid_in[i];
            assign data_out_w[i]  = data_in[i];
            assign ready_in[i]    = ready_out_w[i];
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
