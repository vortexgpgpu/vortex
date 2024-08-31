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
    if (NUM_INPUTS > NUM_OUTPUTS) begin

        wire [NUM_OUTPUTS-1:0][NUM_REQS-1:0]             valid_in_w;
        wire [NUM_OUTPUTS-1:0][NUM_REQS-1:0][DATAW-1:0]  data_in_w;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;
                if (ii < NUM_INPUTS) begin
                    assign valid_in_w[i][j] = valid_in[ii];
                    assign data_in_w[i][j]  = data_in[ii];
                end else begin
                    assign valid_in_w[i][j] = 0;
                    assign data_in_w[i][j]  = '0;
                end
            end
        end

        wire [NUM_OUTPUTS-1:0]            valid_out_w;
        wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out_w;
        wire [NUM_OUTPUTS-1:0]            ready_out_w;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            assign valid_out_w[i] = valid_in_w[i][sel_in[i]];
            assign data_out_w[i]  = data_in_w[i][sel_in[i]];
        end

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;
                if (ii < NUM_INPUTS) begin
                    assign ready_in[ii] = ready_out_w[i] && (sel_in[i] == LOG_NUM_REQS'(j));
                end
            end
        end

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_out_w[i]),
                .ready_in  (ready_out_w[i]),
                .data_in   (data_out_w[i]),
                .data_out  (data_out[i]),
                .valid_out (valid_out[i]),
                .ready_out (ready_out[i])
            );
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin

        wire [NUM_INPUTS-1:0][NUM_REQS-1:0] valid_out_w;
        wire [NUM_INPUTS-1:0][NUM_REQS-1:0] ready_out_w;

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                assign valid_out_w[i][j] = valid_in[i] && (sel_in[i] == LOG_NUM_REQS'(j));
            end
            assign ready_in[i] = ready_out_w[i][sel_in[i]];
        end

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;
                if (ii < NUM_OUTPUTS) begin
                    VX_elastic_buffer #(
                        .DATAW    (DATAW),
                        .SIZE     (`TO_OUT_BUF_SIZE(OUT_BUF)),
                        .OUT_REG  (`TO_OUT_BUF_REG(OUT_BUF))
                    ) out_buf (
                        .clk       (clk),
                        .reset     (reset),
                        .valid_in  (valid_out_w[i][j]),
                        .ready_in  (ready_out_w[i][j]),
                        .data_in   (data_in[i]),
                        .data_out  (data_out[ii]),
                        .valid_out (valid_out[ii]),
                        .ready_out (ready_out[ii])
                    );
                end else begin
                    `UNUSED_VAR (valid_out_w[i][j])
                    assign ready_out_w[i][j] = '0;
                end
            end
        end

    end else begin

        // #Inputs == #Outputs

        `UNUSED_VAR (sel_in)

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            VX_elastic_buffer #(
                .DATAW    (DATAW),
                .SIZE     (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG  (`TO_OUT_BUF_REG(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in[i]),
                .ready_in  (ready_in[i]),
                .data_in   (data_in[i]),
                .data_out  (data_out[i]),
                .valid_out (valid_out[i]),
                .ready_out (ready_out[i])
            );
        end

    end

endmodule
`TRACING_ON
