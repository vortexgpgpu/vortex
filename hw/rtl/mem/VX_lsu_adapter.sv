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

module VX_lsu_adapter import VX_gpu_pkg::*; #(
    parameter NUM_LANES     = 1,
    parameter DATA_SIZE     = 1,
    parameter TAG_WIDTH     = 1,
    parameter TAG_SEL_BITS  = 0,
    parameter `STRING ARBITER = "P",
    parameter REQ_OUT_BUF   = 0,
    parameter RSP_OUT_BUF   = 0
) (
    input wire              clk,
    input wire              reset,

    VX_lsu_mem_if.slave     lsu_mem_if,
    VX_mem_bus_if.master    mem_bus_if [NUM_LANES]
);
    localparam REQ_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE);
    localparam REQ_DATA_WIDTH = 1 + DATA_SIZE + REQ_ADDR_WIDTH + `MEM_REQ_FLAGS_WIDTH + DATA_SIZE * 8;
    localparam RSP_DATA_WIDTH = DATA_SIZE * 8;

    // handle request unpacking

    wire [NUM_LANES-1:0][REQ_DATA_WIDTH-1:0] req_data_in;

    wire [NUM_LANES-1:0] req_valid_out;
    wire [NUM_LANES-1:0][REQ_DATA_WIDTH-1:0] req_data_out;
    wire [NUM_LANES-1:0][TAG_WIDTH-1:0] req_tag_out;
    wire [NUM_LANES-1:0] req_ready_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_req_data_in
        assign req_data_in[i] = {
            lsu_mem_if.req_data.rw,
            lsu_mem_if.req_data.addr[i],
            lsu_mem_if.req_data.data[i],
            lsu_mem_if.req_data.byteen[i],
            lsu_mem_if.req_data.flags[i]
        };
    end

    VX_stream_unpack #(
        .NUM_REQS   (NUM_LANES),
        .DATA_WIDTH (REQ_DATA_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (REQ_OUT_BUF)
    ) stream_unpack (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (lsu_mem_if.req_valid),
        .mask_in    (lsu_mem_if.req_data.mask),
        .data_in    (req_data_in),
        .tag_in     (lsu_mem_if.req_data.tag),
        .ready_in   (lsu_mem_if.req_ready),
        .valid_out  (req_valid_out),
        .data_out   (req_data_out),
        .tag_out    (req_tag_out),
        .ready_out  (req_ready_out)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_bus_req
        assign mem_bus_if[i].req_valid = req_valid_out[i];
        assign {
            mem_bus_if[i].req_data.rw,
            mem_bus_if[i].req_data.addr,
            mem_bus_if[i].req_data.data,
            mem_bus_if[i].req_data.byteen,
            mem_bus_if[i].req_data.flags
         } = req_data_out[i];
        assign mem_bus_if[i].req_data.tag = req_tag_out[i];
        assign req_ready_out[i] = mem_bus_if[i].req_ready;
    end

    // handle response packing

    wire [NUM_LANES-1:0] rsp_valid_out;
    wire [NUM_LANES-1:0][RSP_DATA_WIDTH-1:0] rsp_data_out;
    wire [NUM_LANES-1:0][TAG_WIDTH-1:0] rsp_tag_out;
    wire [NUM_LANES-1:0] rsp_ready_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_bus_rsp
        assign rsp_valid_out[i] = mem_bus_if[i].rsp_valid;
        assign rsp_data_out[i] = mem_bus_if[i].rsp_data.data;
        assign rsp_tag_out[i] = mem_bus_if[i].rsp_data.tag;
        assign mem_bus_if[i].rsp_ready = rsp_ready_out[i];
    end

    VX_stream_pack #(
        .NUM_REQS     (NUM_LANES),
        .DATA_WIDTH   (RSP_DATA_WIDTH),
        .TAG_WIDTH    (TAG_WIDTH),
        .TAG_SEL_BITS (TAG_SEL_BITS),
        .ARBITER      (ARBITER),
        .OUT_BUF      (RSP_OUT_BUF)
    ) stream_pack (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (rsp_valid_out),
        .data_in    (rsp_data_out),
        .tag_in     (rsp_tag_out),
        .ready_in   (rsp_ready_out),
        .valid_out  (lsu_mem_if.rsp_valid),
        .mask_out   (lsu_mem_if.rsp_data.mask),
        .data_out   (lsu_mem_if.rsp_data.data),
        .tag_out    (lsu_mem_if.rsp_data.tag),
        .ready_out  (lsu_mem_if.rsp_ready)
    );

endmodule
