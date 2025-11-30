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

`include "VX_cache_define.vh"

module VX_cache_top import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Number of memory ports
    parameter MEM_PORTS             = 1,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 65536,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64,
    // Number of banks
    parameter NUM_BANKS             = 4,
    // Number of associative ways
    parameter NUM_WAYS              = 4,
    // Size of a word in bytes
    parameter WORD_SIZE             = 16,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 8,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 16,
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 8,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 8,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Enable cache writeback
    parameter WRITEBACK             = 1,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES           = 1,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = 32,

    // Core response output buffer
    parameter CORE_OUT_BUF          = 3,

    // Memory request output buffer
    parameter MEM_OUT_BUF           = 3,

    parameter MEM_TAG_WIDTH         = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, MEM_PORTS, UUID_WIDTH)
 ) (
    input wire clk,
    input wire reset,

// PERF
`ifdef PERF_ENABLE
    output cache_perf_t cache_perf,
`endif

    // Core request
    input  wire                     core_req_valid [NUM_REQS],
    input  wire                     core_req_rw [NUM_REQS],
    input  wire[WORD_SIZE-1:0]      core_req_byteen [NUM_REQS],
    input  wire[`CS_WORD_ADDR_WIDTH-1:0] core_req_addr [NUM_REQS],
    input  wire[`MEM_REQ_FLAGS_WIDTH-1:0] core_req_flags [NUM_REQS],
    input  wire[`CS_WORD_WIDTH-1:0] core_req_data [NUM_REQS],
    input  wire[TAG_WIDTH-1:0]      core_req_tag [NUM_REQS],
    output wire                     core_req_ready [NUM_REQS],

    // Core response
    output wire                     core_rsp_valid [NUM_REQS],
    output wire[`CS_WORD_WIDTH-1:0] core_rsp_data [NUM_REQS],
    output wire[TAG_WIDTH-1:0]      core_rsp_tag [NUM_REQS],
    input  wire                     core_rsp_ready [NUM_REQS],

    // Memory request
    output wire                     mem_req_valid [MEM_PORTS],
    output wire                     mem_req_rw [MEM_PORTS],
    output wire [LINE_SIZE-1:0]     mem_req_byteen [MEM_PORTS],
    output wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_addr [MEM_PORTS],
    output wire [`CS_LINE_WIDTH-1:0] mem_req_data [MEM_PORTS],
    output wire [MEM_TAG_WIDTH-1:0] mem_req_tag [MEM_PORTS],
    input  wire                     mem_req_ready [MEM_PORTS],

    // Memory response
    input  wire                     mem_rsp_valid [MEM_PORTS],
    input  wire [`CS_LINE_WIDTH-1:0] mem_rsp_data [MEM_PORTS],
    input  wire [MEM_TAG_WIDTH-1:0] mem_rsp_tag [MEM_PORTS],
    output wire                     mem_rsp_ready [MEM_PORTS]
);
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_if[MEM_PORTS]();

    // Core request
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_bus_if[i].req_valid = core_req_valid[i];
        assign core_bus_if[i].req_data.rw = core_req_rw[i];
        assign core_bus_if[i].req_data.byteen = core_req_byteen[i];
        assign core_bus_if[i].req_data.addr = core_req_addr[i];
        assign core_bus_if[i].req_data.flags = core_req_flags[i];
        assign core_bus_if[i].req_data.data = core_req_data[i];
        assign core_bus_if[i].req_data.tag = core_req_tag[i];
        assign core_req_ready[i] = core_bus_if[i].req_ready;
    end

    // Core response
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_rsp_valid[i]= core_bus_if[i].rsp_valid;
        assign core_rsp_data[i] = core_bus_if[i].rsp_data.data;
        assign core_rsp_tag[i]  = core_bus_if[i].rsp_data.tag;
        assign core_bus_if[i].rsp_ready = core_rsp_ready[i];
    end

    // Memory request
    for (genvar i = 0; i < MEM_PORTS; ++i) begin
        assign mem_req_valid[i] = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]  = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i]= mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i] = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i] = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i] = mem_bus_if[i].req_data.tag;
        assign mem_bus_if[i].req_ready = mem_req_ready[i];
    end

    // Memory response
    for (genvar i = 0; i < MEM_PORTS; ++i) begin
        assign mem_bus_if[i].rsp_valid = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag = mem_rsp_tag[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

    VX_cache_wrap #(
        .INSTANCE_ID    (INSTANCE_ID),
        .CACHE_SIZE     (CACHE_SIZE),
        .LINE_SIZE      (LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .NUM_WAYS       (NUM_WAYS),
        .WORD_SIZE      (WORD_SIZE),
        .NUM_REQS       (NUM_REQS),
        .MEM_PORTS      (MEM_PORTS),
        .CRSQ_SIZE      (CRSQ_SIZE),
        .MSHR_SIZE      (MSHR_SIZE),
        .MRSQ_SIZE      (MRSQ_SIZE),
        .MREQ_SIZE      (MREQ_SIZE),
        .TAG_WIDTH      (TAG_WIDTH),
        .UUID_WIDTH     (UUID_WIDTH),
        .WRITE_ENABLE   (WRITE_ENABLE),
        .WRITEBACK      (WRITEBACK),
        .DIRTY_BYTES    (DIRTY_BYTES),
        .CORE_OUT_BUF   (CORE_OUT_BUF),
        .MEM_OUT_BUF    (MEM_OUT_BUF)
    ) cache (
    `ifdef PERF_ENABLE
        .cache_perf     (cache_perf),
    `endif
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (core_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

endmodule
