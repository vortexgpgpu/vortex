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
    if (NUM_INPUTS > NUM_OUTPUTS) begin : g_input_select

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf

            wire [NUM_REQS-1:0] valid_in_w;
            wire [NUM_REQS-1:0][DATAW-1:0] data_in_w;
            wire [NUM_REQS-1:0] ready_in_w;

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

            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in_w[sel_in[o]]),
                .ready_in  (ready_in_w[sel_in[o]]),
                .data_in   (data_in_w[sel_in[o]]),
                .data_out  (data_out[o]),
                .valid_out (valid_out[o]),
                .ready_out (ready_out[o])
            );
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin : g_output_select

        // Inputs < Outputs

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_out_buf

            wire [NUM_REQS-1:0] ready_out_w;

            for (genvar r = 0; r < NUM_REQS; ++r) begin : g_r
                localparam o = r * NUM_INPUTS + i;
                if (o < NUM_OUTPUTS) begin : g_valid
                    wire valid_out_w  = valid_in[i] && (sel_in[i] == LOG_NUM_REQS'(r));
                    VX_elastic_buffer #(
                        .DATAW    (DATAW),
                        .SIZE     (`TO_OUT_BUF_SIZE(OUT_BUF)),
                        .OUT_REG  (`TO_OUT_BUF_REG(OUT_BUF))
                    ) out_buf (
                        .clk       (clk),
                        .reset     (reset),
                        .valid_in  (valid_out_w),
                        .ready_in  (ready_out_w[r]),
                        .data_in   (data_in[i]),
                        .data_out  (data_out[o]),
                        .valid_out (valid_out[o]),
                        .ready_out (ready_out[o])
                    );
                end else begin : g_padding
                    assign ready_out_w[r] = '0;
                end
            end

            assign ready_in[i] = ready_out_w[sel_in[i]];
        end

    end else begin : g_passthru

        // #Inputs == #Outputs

        `UNUSED_VAR (sel_in)

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
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
