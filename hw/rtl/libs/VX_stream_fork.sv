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
module VX_stream_fork #(
    parameter NUM_OUTPUTS = 1,
    parameter DATAW       = 1,
    parameter OUT_BUF     = 0,
    parameter EAGER       = 0
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         valid_in,
    input  wire [DATAW-1:0]             data_in,
    output wire                         ready_in,

    output wire [NUM_OUTPUTS-1:0]       valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out,
    input  wire [NUM_OUTPUTS-1:0]       ready_out
);
    if (NUM_OUTPUTS > 1) begin : g_multi_outputs

        if (EAGER != 0) begin : g_eager
            reg [NUM_OUTPUTS-1:0] delivered_r;
            wire [NUM_OUTPUTS-1:0] valid_in_w, ready_in_w;

            assign valid_in_w = {NUM_OUTPUTS{valid_in}} & ~delivered_r;

            wire all_ready = & (delivered_r | ready_in_w);

            always @(posedge clk) begin
                if (reset) begin
                    delivered_r <= '0;
                end else if (valid_in) begin
                    if (all_ready) begin
                        delivered_r <= '0;
                    end else begin
                        delivered_r <= delivered_r | ready_in_w;
                    end
                end
            end

            for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf
                VX_elastic_buffer #(
                    .DATAW   (DATAW),
                    .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                    .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                    .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
                ) out_buf (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in_w[o]),
                    .data_in   (data_in),
                    .ready_in  (ready_in_w[o]),
                    .valid_out (valid_out[o]),
                    .data_out  (data_out[o]),
                    .ready_out (ready_out[o])
                );
            end

            assign ready_in = all_ready;

        end else begin : g_lockstep

            wire [NUM_OUTPUTS-1:0] ready_in_w;
            wire ready_all = (& ready_in_w);

            assign ready_in = ready_all;

            for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_buf
                VX_elastic_buffer #(
                    .DATAW   (DATAW),
                    .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                    .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
                    .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
                ) out_buf (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in && ready_all),
                    .data_in   (data_in),
                    .ready_in  (ready_in_w[o]),
                    .valid_out (valid_out[o]),
                    .data_out  (data_out[o]),
                    .ready_out (ready_out[o])
                );
            end

        end

    end else begin : g_passthru

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_in),
            .data_in   (data_in),
            .ready_in  (ready_in),
            .valid_out (valid_out),
            .data_out  (data_out),
            .ready_out (ready_out)
        );

    end

endmodule
`TRACING_ON
