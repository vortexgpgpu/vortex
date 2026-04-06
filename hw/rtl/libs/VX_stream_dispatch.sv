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

// Pull-based stream dispatcher.
// Arbitrates N inputs into a buffered FIFO, then dispatches to the first
// ready output using a priority encoder.  Unlike VX_stream_switch, the
// output selection is determined by downstream readiness, not by the sender.

`TRACING_OFF
module VX_stream_dispatch #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter DATAW       = 1,
    parameter `STRING ARBITER = "R",
    parameter BUFFERED    = 0,
    parameter OUT_BUF     = 0
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_INPUTS-1:0]              valid_in,
    input  wire [NUM_INPUTS-1:0][DATAW-1:0]   data_in,
    output wire [NUM_INPUTS-1:0]              ready_in,

    output wire [NUM_OUTPUTS-1:0]             valid_out,
    output wire [NUM_OUTPUTS-1:0][DATAW-1:0]  data_out,
    input  wire [NUM_OUTPUTS-1:0]             ready_out
);

    // Stage 1: N-to-1 arbitration

    wire             arb_valid;
    wire [DATAW-1:0] arb_data;
    wire             arb_ready;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_INPUTS),
        .DATAW      (DATAW),
        .ARBITER    (ARBITER)
    ) input_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .data_in   (data_in),
        .ready_in  (ready_in),
        .valid_out (arb_valid),
        .data_out  (arb_data),
        .ready_out (arb_ready),
        `UNUSED_PIN (sel_out)
    );

    // Stage 2: optional deep FIFO

    wire             buf_valid;
    wire [DATAW-1:0] buf_data;
    wire             buf_ready;

    VX_elastic_buffer #(
        .DATAW (DATAW),
        .SIZE  (BUFFERED)
    ) fifo_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (arb_valid),
        .ready_in  (arb_ready),
        .data_in   (arb_data),
        .valid_out (buf_valid),
        .ready_out (buf_ready),
        .data_out  (buf_data)
    );

    // Stage 3: dispatch to first ready output

    // Output demux with optional buffering

    wire [NUM_OUTPUTS-1:0]            valid_out_w;
    wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out_w;
    wire [NUM_OUTPUTS-1:0]            ready_out_w;

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out_buf
        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_out_w[i]),
            .ready_in  (ready_out_w[i]),
            .data_in   (data_out_w[i]),
            .valid_out (valid_out[i]),
            .ready_out (ready_out[i]),
            .data_out  (data_out[i])
        );
    end

    // Priority encoder selects first ready output (buffer input-side ready)

    wire [NUM_OUTPUTS-1:0] sel_onehot;

    VX_priority_encoder #(
        .N (NUM_OUTPUTS)
    ) idle_sel (
        .data_in   (ready_out_w),
        .onehot_out(sel_onehot),
        `UNUSED_PIN (index_out),
        .valid_out (buf_ready)
    );

    wire dispatch_fire = buf_valid && buf_ready;

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_out
        assign valid_out_w[i] = dispatch_fire && sel_onehot[i];
        assign data_out_w[i]  = buf_data;
    end

endmodule
`TRACING_ON
