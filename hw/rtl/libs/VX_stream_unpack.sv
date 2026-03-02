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
module VX_stream_unpack #(
    parameter NUM_REQS      = 1,
    parameter DATA_WIDTH    = 1,
    parameter TAG_WIDTH     = 1,
    parameter OUT_BUF       = 0
) (
    input wire                          clk,
    input wire                          reset,

    // input
    input wire                          valid_in,
    input wire [NUM_REQS-1:0]           mask_in,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0] data_in,
    input wire [TAG_WIDTH-1:0]          tag_in,
    output wire                         ready_in,

    // output
    output wire [NUM_REQS-1:0]          valid_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] data_out,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0] tag_out,
    input wire  [NUM_REQS-1:0]          ready_out
);
    if (NUM_REQS > 1) begin : g_unpack

        reg [NUM_REQS-1:0] delivered_r;
        wire [NUM_REQS-1:0] valid_in_w, ready_in_w;

        assign valid_in_w = {NUM_REQS{valid_in}} & mask_in & ~delivered_r;

        wire all_ready = & (~mask_in | delivered_r | ready_in_w);

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

        assign ready_in = all_ready;

        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_outbuf
            VX_elastic_buffer #(
                .DATAW   (DATA_WIDTH + TAG_WIDTH),
                .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
            ) out_buf (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_in_w[i]),
                .ready_in  (ready_in_w[i]),
                .data_in   ({data_in[i],  tag_in}),
                .data_out  ({data_out[i], tag_out[i]}),
                .valid_out (valid_out[i]),
                .ready_out (ready_out[i])
            );
        end

    end else begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (mask_in)
        assign valid_out = valid_in;
        assign data_out  = data_in;
        assign tag_out   = tag_in;
        assign ready_in  = ready_out;

    end

endmodule
`TRACING_ON
