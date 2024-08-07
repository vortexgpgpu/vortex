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

module VX_mem_arb #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter DATA_SIZE      = 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH     = (MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE)),
    parameter TAG_WIDTH      = 1,
    parameter TAG_SEL_IDX    = 0,
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_mem_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_mem_bus_if.master    bus_out_if [NUM_OUTPUTS]
);
    localparam DATA_WIDTH   = (8 * DATA_SIZE);
    localparam LOG_NUM_REQS = `ARB_SEL_BITS(NUM_INPUTS, NUM_OUTPUTS);
    localparam REQ_DATAW    = TAG_WIDTH + ADDR_WIDTH + `MEM_REQ_FLAGS_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW    = TAG_WIDTH + DATA_WIDTH;

    `STATIC_ASSERT ((NUM_INPUTS >= NUM_OUTPUTS), ("invalid parameter"))

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0][`UP(LOG_NUM_REQS)-1:0] req_sel_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i] = {
            bus_in_if[i].req_data.rw,
            bus_in_if[i].req_data.byteen,
            bus_in_if[i].req_data.addr,
            bus_in_if[i].req_data.flags,
            bus_in_if[i].req_data.data,
            bus_in_if[i].req_data.tag
        };
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
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

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
        wire [TAG_WIDTH-1:0] req_tag_out;
        VX_bits_insert #(
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_insert (
            .data_in  (req_tag_out),
            .ins_in   (req_sel_out[i]),
            .data_out (bus_out_if[i].req_data.tag)
        );
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign {
            bus_out_if[i].req_data.rw,
            bus_out_if[i].req_data.byteen,
            bus_out_if[i].req_data.addr,
            bus_out_if[i].req_data.flags,
            bus_out_if[i].req_data.data,
            req_tag_out
        } = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_INPUTS-1:0]                 rsp_valid_out;
    wire [NUM_INPUTS-1:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [NUM_INPUTS-1:0]                 rsp_ready_out;

    wire [NUM_OUTPUTS-1:0]                rsp_valid_in;
    wire [NUM_OUTPUTS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_OUTPUTS-1:0]                rsp_ready_in;

    if (NUM_INPUTS > NUM_OUTPUTS) begin

        wire [NUM_OUTPUTS-1:0][LOG_NUM_REQS-1:0] rsp_sel_in;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            wire [TAG_WIDTH-1:0] rsp_tag_out;
            VX_bits_remove #(
                .N   (TAG_WIDTH + LOG_NUM_REQS),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (bus_out_if[i].rsp_data.tag),
                .data_out (rsp_tag_out)
            );

            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i] = {rsp_tag_out, bus_out_if[i].rsp_data.data};
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];

            if (NUM_INPUTS > 1) begin
                assign rsp_sel_in[i] = bus_out_if[i].rsp_data.tag[TAG_SEL_IDX +: LOG_NUM_REQS];
            end else begin
                assign rsp_sel_in[i] = '0;
            end
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

    end else begin

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i] = {
                bus_out_if[i].rsp_data.tag,
                bus_out_if[i].rsp_data.data
            };
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
        end

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_OUTPUTS),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (RSP_OUT_BUF)
        ) req_arb (
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

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
        assign {
            bus_in_if[i].rsp_data.tag,
            bus_in_if[i].rsp_data.data
        } = rsp_data_out[i];
        assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
    end

endmodule
