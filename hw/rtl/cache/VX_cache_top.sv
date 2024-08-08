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

    // Size of cache in bytes
    parameter CACHE_SIZE            = 16384,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64,
    // Number of banks
    parameter NUM_BANKS             = 4,
    // Number of associative ways
    parameter NUM_WAYS              = 4,
    // Size of a word in bytes
    parameter WORD_SIZE             = 4,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 16,
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Enable cache writeback
    parameter WRITEBACK             = 0,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES           = 0,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = 16,

    // Core response output buffer
    parameter CORE_OUT_BUF          = 2,

    // Memory request output buffer
    parameter MEM_OUT_BUF           = 2,

    parameter MEM_TAG_WIDTH = `CLOG2(MSHR_SIZE) + `CLOG2(NUM_BANKS)
 ) (
    input wire clk,
    input wire reset,

// PERF
`ifdef PERF_ENABLE
    output cache_perf_t cache_perf,
`endif

    // Core request
    input  wire [NUM_REQS-1:0]                 core_req_valid,
    input  wire [NUM_REQS-1:0]                 core_req_rw,
    input  wire [NUM_REQS-1:0][WORD_SIZE-1:0]  core_req_byteen,
    input  wire [NUM_REQS-1:0][`CS_WORD_ADDR_WIDTH-1:0] core_req_addr,
    input  wire [NUM_REQS-1:0][`ADDR_TYPE_WIDTH-1:0] core_req_atype,
    input  wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_req_data,
    input  wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  core_req_tag,
    output wire [NUM_REQS-1:0]                 core_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                 core_rsp_valid,
    output wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  core_rsp_tag,
    input  wire [NUM_REQS-1:0]                 core_rsp_ready,

    // Memory request
    output wire                    mem_req_valid,
    output wire                    mem_req_rw,
    output wire [LINE_SIZE-1:0]    mem_req_byteen,
    output wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [`CS_LINE_WIDTH-1:0] mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0] mem_req_tag,
    input  wire                    mem_req_ready,

    // Memory response
    input  wire                    mem_rsp_valid,
    input  wire [`CS_LINE_WIDTH-1:0] mem_rsp_data,
    input  wire [MEM_TAG_WIDTH-1:0] mem_rsp_tag,
    output wire                    mem_rsp_ready
);
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_if();

    // Core request
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_bus_if[i].req_valid = core_req_valid[i];
        assign core_bus_if[i].req_data.rw = core_req_rw[i];
        assign core_bus_if[i].req_data.byteen = core_req_byteen[i];
        assign core_bus_if[i].req_data.addr = core_req_addr[i];
        assign core_bus_if[i].req_data.atype = core_req_atype[i];
        assign core_bus_if[i].req_data.data = core_req_data[i];
        assign core_bus_if[i].req_data.tag = core_req_tag[i];
        assign core_req_ready[i] = core_bus_if[i].req_ready;
    end

    // Core response
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_rsp_valid[i] = core_bus_if[i].rsp_valid;
        assign core_rsp_data[i] = core_bus_if[i].rsp_data.data;
        assign core_rsp_tag[i] = core_bus_if[i].rsp_data.tag;
        assign core_bus_if[i].rsp_ready = core_rsp_ready[i];
    end

    // Memory request
    assign mem_req_valid = mem_bus_if.req_valid;
    assign mem_req_rw = mem_bus_if.req_data.rw;
    assign mem_req_byteen = mem_bus_if.req_data.byteen;
    assign mem_req_addr = mem_bus_if.req_data.addr;
    assign mem_req_data = mem_bus_if.req_data.data;
    assign mem_req_tag = mem_bus_if.req_data.tag;
    assign mem_bus_if.req_ready = mem_req_ready;
    `UNUSED_VAR (mem_bus_if.req_data.atype)

    // Memory response
    assign mem_bus_if.rsp_valid = mem_rsp_valid;
    assign mem_bus_if.rsp_data.data = mem_rsp_data;
    assign mem_bus_if.rsp_data.tag = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

    VX_cache #(
        .INSTANCE_ID    (INSTANCE_ID),
        .CACHE_SIZE     (CACHE_SIZE),
        .LINE_SIZE      (LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .NUM_WAYS       (NUM_WAYS),
        .WORD_SIZE      (WORD_SIZE),
        .NUM_REQS       (NUM_REQS),
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
