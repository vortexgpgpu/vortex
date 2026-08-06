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

// ============================================================================
// VX_mem_bus_slice — point-to-point register slice for a VX_mem_bus_if.
//
// Inserts an elastic-buffer stage on the request and/or response channel of a
// single memory bus, for a clean producer/consumer boundary (e.g. terminating
// a long addr-gen cone or SLR-crossing route at a flop). REQ_OUT_BUF /
// RSP_OUT_BUF use the standard TO_OUT_BUF encoding (0 = passthrough).
// ============================================================================

`TRACING_OFF
module VX_mem_bus_slice import VX_gpu_pkg::*; #(
    parameter DATA_SIZE   = 1,
    parameter TAG_WIDTH   = 1,
    parameter REQ_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0,
    parameter ADDR_WIDTH  = (`VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)),
    parameter ATTR_WIDTH  = MEM_ATTR_WIDTH
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     bus_in_if,
    VX_mem_bus_if.master    bus_out_if
);
    localparam DATA_WIDTH = (8 * DATA_SIZE);
    localparam REQ_DATAW  = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + ATTR_WIDTH + TAG_WIDTH;
    localparam RSP_DATAW  = DATA_WIDTH + TAG_WIDTH;

    // ---- Request : bus_in -> bus_out ----
    wire [REQ_DATAW-1:0] req_data_in = bus_in_if.req_data;
    wire [REQ_DATAW-1:0] req_data_out;

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(REQ_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(REQ_OUT_BUF))
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid),
        .ready_in  (bus_in_if.req_ready),
        .data_in   (req_data_in),
        .data_out  (req_data_out),
        .valid_out (bus_out_if.req_valid),
        .ready_out (bus_out_if.req_ready)
    );
    assign bus_out_if.req_data = req_data_out;

    // ---- Response : bus_out -> bus_in ----
    wire [RSP_DATAW-1:0] rsp_data_in = bus_out_if.rsp_data;
    wire [RSP_DATAW-1:0] rsp_data_out;

    VX_elastic_buffer #(
        .DATAW   (RSP_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(RSP_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(RSP_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(RSP_OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_out_if.rsp_valid),
        .ready_in  (bus_out_if.rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (rsp_data_out),
        .valid_out (bus_in_if.rsp_valid),
        .ready_out (bus_in_if.rsp_ready)
    );
    assign bus_in_if.rsp_data = rsp_data_out;

endmodule
`TRACING_ON
