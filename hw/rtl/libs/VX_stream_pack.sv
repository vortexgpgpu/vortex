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
module VX_stream_pack #(
    parameter NUM_REQS      = 1,
    parameter DATA_WIDTH    = 1,
    parameter TAG_WIDTH     = 1,
    parameter TAG_SEL_BITS  = 0,
    parameter `STRING ARBITER = "P",
    parameter OUT_BUF       = 0
) (
    input wire                          clk,
    input wire                          reset,

    // input
    input wire [NUM_REQS-1:0]           valid_in,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0] data_in,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0] tag_in,
    output wire [NUM_REQS-1:0]          ready_in,

    // output
    output wire                         valid_out,
    output wire [NUM_REQS-1:0]          mask_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] data_out,
    output wire [TAG_WIDTH-1:0]         tag_out,
    input wire                          ready_out
);
    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0] grant_onehot;
        wire grant_valid;
        wire grant_ready;

        VX_generic_arbiter #(
            .NUM_REQS (NUM_REQS),
            .TYPE     (ARBITER)
        ) arbiter (
            .clk         (clk),
            .reset       (reset),
            .requests    (valid_in),
            .grant_valid (grant_valid),
            `UNUSED_PIN  (grant_index),
            .grant_onehot(grant_onehot),
            .grant_ready (grant_ready)
        );

        wire [TAG_WIDTH-1:0] tag_sel;

        VX_onehot_mux #(
            .DATAW (TAG_WIDTH),
            .N     (NUM_REQS)
        ) onehot_mux (
            .data_in  (tag_in),
            .sel_in   (grant_onehot),
            .data_out (tag_sel)
        );

        wire [NUM_REQS-1:0] tag_matches;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign tag_matches[i] = (tag_in[i][TAG_SEL_BITS-1:0] == tag_sel[TAG_SEL_BITS-1:0]);
        end

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign ready_in[i] = grant_ready & tag_matches[i];
        end

        wire [NUM_REQS-1:0] mask_sel = valid_in & tag_matches;

        VX_elastic_buffer #(
            .DATAW   (NUM_REQS + TAG_WIDTH + (NUM_REQS * DATA_WIDTH)),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (grant_valid),
            .data_in   ({mask_sel, tag_sel, data_in}),
            .ready_in  (grant_ready),
            .valid_out (valid_out),
            .data_out  ({mask_out, tag_out, data_out}),
            .ready_out (ready_out)
        );

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign valid_out = valid_in;
        assign mask_out  = 1'b1;
        assign data_out  = data_in;
        assign tag_out   = tag_in;
        assign ready_in  = ready_out;

    end

endmodule
`TRACING_ON
