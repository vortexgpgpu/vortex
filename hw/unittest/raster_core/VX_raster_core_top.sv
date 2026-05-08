//!/bin/bash

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

`include "VX_raster_define.vh"

module VX_raster_core_top import VX_gpu_pkg::*; import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
    parameter NUM_SLICES      = 1, // number of slices
    parameter TILE_LOGSIZE    = 5, // tile log size
    parameter BLOCK_LOGSIZE   = 2, // block log size
    parameter MEM_FIFO_DEPTH  = 8, // memory queue size
    parameter QUAD_FIFO_DEPTH = 8, // quad queue size
    parameter OUTPUT_QUADS    = 4   // number of output quads
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    output wire                             raster_req_valid,
    output raster_stamp_t [OUTPUT_QUADS-1:0] raster_req_stamps,
    output wire                             raster_req_done,
    input wire                              raster_req_ready,

    output wire [RCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [RCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [RCACHE_NUM_REQS-1:0][RCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [RCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [RCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [RCACHE_NUM_REQS-1:0][RCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [RCACHE_NUM_REQS-1:0][RCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [RCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_raster_perf_if perf_raster_if();
`ifndef PERF_ENABLE
    assign perf_raster_if.mem_reads    = '0;
    assign perf_raster_if.mem_latency  = '0;
    assign perf_raster_if.stall_cycles = '0;
    `UNUSED_VAR (perf_raster_if.mem_reads)
    `UNUSED_VAR (perf_raster_if.mem_latency)
    `UNUSED_VAR (perf_raster_if.stall_cycles)
`endif

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.req_valid     = dcr_write_valid;
    assign dcr_bus_if.req_data.rw   = 1'b1;
    assign dcr_bus_if.req_data.addr = dcr_write_addr;
    assign dcr_bus_if.req_data.data = dcr_write_data;
    `UNUSED_VAR (dcr_bus_if.rsp_valid)
    `UNUSED_VAR (dcr_bus_if.rsp_data)

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) raster_bus_if();

    assign raster_req_valid = raster_bus_if.req_valid;
    assign raster_req_stamps = raster_bus_if.req_data.stamps;
    assign raster_req_done = raster_bus_if.req_data.done;
    assign raster_bus_if.req_ready = raster_req_ready;

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) cache_bus_if[RCACHE_NUM_REQS]();

    assign cache_req_valid = cache_bus_if[0].req_valid;
    assign cache_req_rw = cache_bus_if[0].req_data.rw;
    assign cache_req_byteen = cache_bus_if[0].req_data.byteen;
    assign cache_req_addr = cache_bus_if[0].req_data.addr;
    assign cache_req_data = cache_bus_if[0].req_data.data;
    assign cache_req_tag = cache_bus_if[0].req_data.tag;
    assign cache_bus_if[0].req_ready = cache_req_ready;
    `UNUSED_VAR (cache_bus_if[0].req_data.attr)

    assign cache_bus_if[0].rsp_valid = cache_rsp_valid;
    assign cache_bus_if[0].rsp_data.tag = cache_rsp_tag;
    assign cache_bus_if[0].rsp_data.data = cache_rsp_data;
    assign cache_rsp_ready = cache_bus_if[0].rsp_ready;

    VX_raster_core #(
        .INSTANCE_ID     (INSTANCE_ID),
        .INSTANCE_IDX    (INSTANCE_IDX),
        .NUM_INSTANCES   (NUM_INSTANCES),
        .NUM_SLICES      (NUM_SLICES),
        .TILE_LOGSIZE    (TILE_LOGSIZE),
        .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
        .MEM_FIFO_DEPTH  (MEM_FIFO_DEPTH),
        .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH),
        .OUTPUT_QUADS    (OUTPUT_QUADS)
    ) raster_core (
        `SCOPE_IO_BIND (0)
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_raster_if(perf_raster_if),
    `endif
        .dcr_bus_if    (dcr_bus_if),
        .raster_bus_if (raster_bus_if),
        .cache_bus_if  (cache_bus_if)
    );

endmodule
