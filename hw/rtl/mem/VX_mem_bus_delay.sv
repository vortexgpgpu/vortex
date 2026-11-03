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

// A memory link with a fixed latency in each direction: requests and
// responses each take LATENCY cycles to cross it, at full throughput.
`TRACING_OFF
module VX_mem_bus_delay import VX_gpu_pkg::*; #(
    parameter DATA_SIZE  = 1,
    parameter TAG_WIDTH  = 1,
    parameter LATENCY    = 1,
    parameter ADDR_WIDTH = (`VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)),
    parameter ATTR_WIDTH = MEM_ATTR_WIDTH
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     bus_in_if,
    VX_mem_bus_if.master    bus_out_if,
    output wire             empty
);
    localparam DATA_WIDTH = (8 * DATA_SIZE);
    localparam REQ_DATAW  = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH;
    localparam RSP_DATAW  = DATA_WIDTH + TAG_WIDTH;

    wire req_empty, rsp_empty;

    wire [REQ_DATAW-1:0] req_data_in = bus_in_if.req_data;
    wire [REQ_DATAW-1:0] req_data_out;

    VX_delay_buffer #(
        .DATAW   (REQ_DATAW),
        .LATENCY (LATENCY)
    ) req_delay (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid),
        .ready_in  (bus_in_if.req_ready),
        .data_in   (req_data_in),
        .data_out  (req_data_out),
        .valid_out (bus_out_if.req_valid),
        .ready_out (bus_out_if.req_ready),
        .empty     (req_empty)
    );
    assign bus_out_if.req_data = req_data_out;

    wire [RSP_DATAW-1:0] rsp_data_in = bus_out_if.rsp_data;
    wire [RSP_DATAW-1:0] rsp_data_out;

    VX_delay_buffer #(
        .DATAW   (RSP_DATAW),
        .LATENCY (LATENCY)
    ) rsp_delay (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_out_if.rsp_valid),
        .ready_in  (bus_out_if.rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (rsp_data_out),
        .valid_out (bus_in_if.rsp_valid),
        .ready_out (bus_in_if.rsp_ready),
        .empty     (rsp_empty)
    );
    assign bus_in_if.rsp_data = rsp_data_out;

    assign empty = req_empty && rsp_empty;

endmodule
`TRACING_ON
