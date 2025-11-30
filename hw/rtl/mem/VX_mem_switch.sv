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

module VX_mem_switch import VX_gpu_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter DATA_SIZE      = 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH     = (MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE)),
    parameter TAG_WIDTH      = 1,
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter `STRING ARBITER = "R",
    parameter NUM_REQS       = (NUM_INPUTS > NUM_OUTPUTS) ? `CDIV(NUM_INPUTS, NUM_OUTPUTS) : `CDIV(NUM_OUTPUTS, NUM_INPUTS),
    parameter SEL_COUNT      = `MIN(NUM_INPUTS, NUM_OUTPUTS),
    parameter LOG_NUM_REQS   = `CLOG2(NUM_REQS)
) (
    input wire              clk,
    input wire              reset,

    input wire  [SEL_COUNT-1:0][`UP(LOG_NUM_REQS)-1:0] bus_sel,
    VX_mem_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_mem_bus_if.master    bus_out_if [NUM_OUTPUTS]
);
    localparam DATA_WIDTH = (8 * DATA_SIZE);
    localparam REQ_DATAW  = TAG_WIDTH + ADDR_WIDTH + `MEM_REQ_FLAGS_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW  = TAG_WIDTH + DATA_WIDTH;

    // handle requests ////////////////////////////////////////////////////////

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i]  = bus_in_if[i].req_data;
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_switch #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (REQ_DATAW),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (bus_sel),
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .ready_in  (req_ready_in),
        .valid_out (req_valid_out),
        .data_out  (req_data_out),
        .ready_out (req_ready_out)
    );

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_req_data_out
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign bus_out_if[i].req_data  = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

    // handle responses ///////////////////////////////////////////////////////

    wire [NUM_OUTPUTS-1:0]              rsp_valid_in;
    wire [NUM_OUTPUTS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_OUTPUTS-1:0]              rsp_ready_in;

    wire [NUM_INPUTS-1:0]               rsp_valid_out;
    wire [NUM_INPUTS-1:0][RSP_DATAW-1:0] rsp_data_out;
    wire [NUM_INPUTS-1:0]               rsp_ready_out;

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_rsp_data_in
        assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
        assign rsp_data_in[i]  = bus_out_if[i].rsp_data;
        assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_OUTPUTS),
        .NUM_OUTPUTS(NUM_INPUTS),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_in),
        .data_in   (rsp_data_in),
        .ready_in  (rsp_ready_in),
        .valid_out (rsp_valid_out),
        .data_out  (rsp_data_out),
        .ready_out (rsp_ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_rsp_data_out
        assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
        assign bus_in_if[i].rsp_data  = rsp_data_out[i];
        assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
    end

endmodule
