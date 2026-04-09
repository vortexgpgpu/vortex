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

// Memory crossbar with per-input routing selection.
// Uses VX_stream_xbar for request-side N:M routing with explicit sel_in.
// Response routing uses tag-embedded selection bits (like VX_mem_arb).

module VX_mem_xbar import VX_gpu_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter DATA_SIZE      = 1,
    parameter TAG_WIDTH      = 1,
    parameter TAG_SEL_IDX    = 0,
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter `STRING ARBITER = "R",
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH     = (MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE)),
    parameter FLAGS_WIDTH    = MEM_FLAGS_WIDTH
) (
    input wire              clk,
    input wire              reset,

    input wire [NUM_INPUTS-1:0][`UP(`CLOG2(NUM_OUTPUTS))-1:0] sel_in,
    VX_mem_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_mem_bus_if.master    bus_out_if [NUM_OUTPUTS]
);
    `UNUSED_PARAM (TAG_WIDTH)
    localparam DATA_WIDTH   = (8 * DATA_SIZE);
    localparam LOG_NUM_REQS = `ARB_SEL_BITS(NUM_INPUTS, NUM_OUTPUTS);
    localparam TAG_UUID_W   = `UP(UUID_WIDTH);
    localparam REQ_DATAW    = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + FLAGS_WIDTH + TAG_UUID_W;
    localparam RSP_DATAW    = DATA_WIDTH + TAG_UUID_W;
    localparam SEL_COUNT    = `MIN(NUM_INPUTS, NUM_OUTPUTS);

    // handle requests ////////////////////////////////////////////////////////

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [SEL_COUNT-1:0][`UP(`CLOG2(NUM_INPUTS))-1:0] req_sel_out;
    `IGNORE_UNOPTFLAT_BEGIN
    wire [NUM_OUTPUTS-1:0]                req_ready_out;
    `IGNORE_UNOPTFLAT_END

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i] = REQ_DATAW'({
            bus_in_if[i].req_data.rw,
            bus_in_if[i].req_data.addr,
            bus_in_if[i].req_data.data,
            bus_in_if[i].req_data.byteen,
            bus_in_if[i].req_data.flags,
            bus_in_if[i].req_data.tag.uuid
        });
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (REQ_DATAW),
        .ARBITER     (ARBITER),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .ready_in  (req_ready_in),
        .data_in   (req_data_in),
        .sel_in    (sel_in),
        .data_out  (req_data_out),
        .sel_out   (req_sel_out),
        .valid_out (req_valid_out),
        .ready_out (req_ready_out),
        `UNUSED_PIN (collisions)
    );

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_bus_out_if
        wire [TAG_UUID_W-1:0] req_tag_out;
        `UNUSED_VAR (req_tag_out)
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign {
            bus_out_if[i].req_data.rw,
            bus_out_if[i].req_data.addr,
            bus_out_if[i].req_data.data,
            bus_out_if[i].req_data.byteen,
            bus_out_if[i].req_data.flags,
            req_tag_out
        } = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;

        if (NUM_INPUTS > NUM_OUTPUTS) begin : g_req_tag_sel_out
            wire [TAG_UUID_W+LOG_NUM_REQS-1:0] req_tag_wide;
            VX_bits_insert #(
                .N   (TAG_UUID_W),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_insert (
                .data_in  (req_tag_out),
                .ins_in   (req_sel_out[i]),
                .data_out (req_tag_wide)
            );
            assign bus_out_if[i].req_data.tag.uuid = req_tag_wide[TAG_UUID_W-1:0];
            assign bus_out_if[i].req_data.tag.value = '0;
        end else begin : g_req_tag_out
            `UNUSED_VAR (req_sel_out)
            assign bus_out_if[i].req_data.tag.uuid = req_tag_out[TAG_UUID_W-1:0];
            assign bus_out_if[i].req_data.tag.value = '0;
        end
    end

    // handle responses ///////////////////////////////////////////////////////

    wire [NUM_INPUTS-1:0]                 rsp_valid_out;
    wire [NUM_INPUTS-1:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [NUM_INPUTS-1:0]                 rsp_ready_out;

    wire [NUM_OUTPUTS-1:0]                rsp_valid_in;
    wire [NUM_OUTPUTS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_OUTPUTS-1:0]                rsp_ready_in;

    if (NUM_INPUTS > NUM_OUTPUTS) begin : g_rsp_select

        wire [NUM_OUTPUTS-1:0][LOG_NUM_REQS-1:0] rsp_sel_in;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_rsp_data_in
            wire [TAG_UUID_W-1:0] rsp_tag_out;
            VX_bits_remove #(
                .N   (TAG_UUID_W + LOG_NUM_REQS),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (bus_out_if[i].rsp_data.tag.uuid),
                .sel_out  (rsp_sel_in[i]),
                .data_out (rsp_tag_out)
            );
            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i]  = RSP_DATAW'({bus_out_if[i].rsp_data.data, rsp_tag_out});
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
        end

        VX_stream_switch #(
            .NUM_INPUTS  (NUM_OUTPUTS),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .OUT_BUF     (RSP_OUT_BUF)
        ) rsp_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel_in),
            .valid_in  (rsp_valid_in),
            .ready_in  (rsp_ready_in),
            .data_in   (rsp_data_in),
            .data_out  (rsp_data_out),
            .valid_out (rsp_valid_out),
            .ready_out (rsp_ready_out)
        );

    end else begin : g_rsp_arb

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_rsp_data_in
            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i] = RSP_DATAW'({
                bus_out_if[i].rsp_data.data,
                bus_out_if[i].rsp_data.tag.uuid
            });
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
        end

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_OUTPUTS),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (RSP_OUT_BUF)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid_in),
            .ready_in  (rsp_ready_in),
            .data_in   (rsp_data_in),
            .data_out  (rsp_data_out),
            .valid_out (rsp_valid_out),
            .ready_out (rsp_ready_out),
            `UNUSED_PIN (sel_out)
        );

    end

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_output
        wire [TAG_UUID_W-1:0] rsp_tag_out;
        `UNUSED_VAR (rsp_tag_out)
        wire [DATA_WIDTH-1:0] rsp_data_dat;
        assign {rsp_data_dat, rsp_tag_out} = rsp_data_out[i];
        assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
        assign bus_in_if[i].rsp_data.data = rsp_data_dat;
        assign bus_in_if[i].rsp_data.tag.uuid = rsp_tag_out[TAG_UUID_W-1:0];
        assign bus_in_if[i].rsp_data.tag.value = '0;
        assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
    end

endmodule
