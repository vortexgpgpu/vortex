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

`include "VX_define.vh"

// N-to-1 arbiter for the PTW miss/fill bus. The input index is folded into
// the low tag bits on the way up and stripped on the way back, so every
// level of the core/socket/cluster hierarchy can stack one of these.
module VX_ptw_arb import VX_gpu_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter TAG_WIDTH   = 1,
    parameter REQ_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_ptw_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_ptw_bus_if.master    bus_out_if
);
    localparam LOG_NUM_REQS  = `ARB_SEL_BITS(NUM_INPUTS, 1);
    localparam TAG_WIDTH_OUT = TAG_WIDTH + LOG_NUM_REQS;
    localparam REQ_DATAW     = VM_VPN_WIDTH + VM_PPN_WIDTH + TAG_WIDTH;
    localparam RSP_DATAW     = VM_PPN_WIDTH + VM_LEVEL_BITS + VM_PTE_FLAGS_WIDTH + 1 + TAG_WIDTH;

    // request path: arbitrate and stamp the source index into the tag

    wire [NUM_INPUTS-1:0]                req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0] req_data_in;
    wire [NUM_INPUTS-1:0]                req_ready_in;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i]  = bus_in_if[i].req_data;
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    wire                          req_valid_out;
    wire [REQ_DATAW-1:0]          req_data_out;
    wire [`UP(LOG_NUM_REQS)-1:0]  req_sel_out;
    wire                          req_ready_out;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (1),
        .DATAW       (REQ_DATAW),
        .ARBITER     (ARBITER),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .ready_in  (req_ready_in),
        .data_in   (req_data_in),
        .data_out  (req_data_out),
        .sel_out   (req_sel_out),
        .valid_out (req_valid_out),
        .ready_out (req_ready_out)
    );

    wire [TAG_WIDTH-1:0] req_tag_out;
    assign bus_out_if.req_valid = req_valid_out;
    assign {
        bus_out_if.req_data.vpn,
        bus_out_if.req_data.root_ppn,
        req_tag_out
    } = req_data_out;
    assign req_ready_out = bus_out_if.req_ready;

    if (NUM_INPUTS > 1) begin : g_req_tag_sel
        VX_bits_insert #(
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (0)
        ) bits_insert (
            .data_in  (req_tag_out),
            .ins_in   (req_sel_out),
            .data_out (bus_out_if.req_data.tag)
        );
    end else begin : g_req_tag
        `UNUSED_VAR (req_sel_out)
        assign bus_out_if.req_data.tag = req_tag_out;
    end

    // response path: strip the source index and route back

    wire [NUM_INPUTS-1:0]                rsp_valid_out;
    wire [NUM_INPUTS-1:0][RSP_DATAW-1:0] rsp_data_out;
    wire [NUM_INPUTS-1:0]                rsp_ready_out;

    if (NUM_INPUTS > 1) begin : g_rsp_switch

        wire [LOG_NUM_REQS-1:0] rsp_sel_in;
        wire [TAG_WIDTH-1:0]    rsp_tag_in;

        VX_bits_remove #(
            .N   (TAG_WIDTH_OUT),
            .S   (LOG_NUM_REQS),
            .POS (0)
        ) bits_remove (
            .data_in  (bus_out_if.rsp_data.tag),
            .sel_out  (rsp_sel_in),
            .data_out (rsp_tag_in)
        );

        wire [RSP_DATAW-1:0] rsp_data_in = {
            bus_out_if.rsp_data.ppn,
            bus_out_if.rsp_data.level,
            bus_out_if.rsp_data.flags,
            bus_out_if.rsp_data.fault,
            rsp_tag_in
        };

        VX_stream_switch #(
            .NUM_INPUTS  (1),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .OUT_BUF     (RSP_OUT_BUF)
        ) rsp_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel_in),
            .valid_in  (bus_out_if.rsp_valid),
            .ready_in  (bus_out_if.rsp_ready),
            .data_in   (rsp_data_in),
            .data_out  (rsp_data_out),
            .valid_out (rsp_valid_out),
            .ready_out (rsp_ready_out)
        );

    end else begin : g_rsp_passthru

        assign rsp_valid_out[0]   = bus_out_if.rsp_valid;
        assign rsp_data_out[0]    = bus_out_if.rsp_data;
        assign bus_out_if.rsp_ready = rsp_ready_out[0];

    end

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_bus_in_if
        assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
        assign bus_in_if[i].rsp_data  = rsp_data_out[i];
        assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
    end

endmodule
