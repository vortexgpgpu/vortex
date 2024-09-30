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
    parameter NUM_REQS       = 1,
    parameter DATA_SIZE      = 1,
    parameter TAG_WIDTH      = 1,
    parameter ADDR_WIDTH     = 1,
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter `STRING ARBITER = "R",
    parameter LOG_NUM_REQS   = `CLOG2(NUM_REQS)
) (
    input wire              clk,
    input wire              reset,

    input wire [`UP(LOG_NUM_REQS)-1:0] bus_sel,
    VX_mem_bus_if.slave     bus_in_if,
    VX_mem_bus_if.master    bus_out_if [NUM_REQS]
);
    localparam DATA_WIDTH = (8 * DATA_SIZE);
    localparam REQ_DATAW  = TAG_WIDTH + ADDR_WIDTH + `MEM_REQ_FLAGS_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW  = TAG_WIDTH + DATA_WIDTH;

    // handle requests ////////////////////////////////////////////////////////

    wire [NUM_REQS-1:0]                req_valid_out;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_REQS-1:0]                req_ready_out;

    VX_stream_switch #(
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW),
        .OUT_BUF     (REQ_OUT_BUF)
    ) req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (bus_sel),
        .valid_in  (bus_in_if.req_valid),
        .data_in   (bus_in_if.req_data),
        .ready_in  (bus_in_if.req_ready),
        .valid_out (req_valid_out),
        .data_out  (req_data_out),
        .ready_out (req_ready_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign bus_out_if[i].req_data = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

    // handle responses ///////////////////////////////////////////////////////

    wire [NUM_REQS-1:0]              rsp_valid_in;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_REQS-1:0]              rsp_ready_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
        assign rsp_data_in[i] = bus_out_if[i].rsp_data;
        assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_REQS),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_in),
        .data_in   (rsp_data_in),
        .ready_in  (rsp_ready_in),
        .valid_out (bus_in_if.rsp_valid),
        .data_out  (bus_in_if.rsp_data),
        .ready_out (bus_in_if.rsp_ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
