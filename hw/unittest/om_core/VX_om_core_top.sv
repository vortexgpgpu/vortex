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

`include "VX_om_define.vh"

module VX_om_core_top import VX_gpu_pkg::*; import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `NUM_THREADS
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    input  wire                             om_req_valid,
    input  wire [UUID_WIDTH-1:0]           om_req_uuid,
    input  wire [NUM_LANES-1:0]             om_req_mask,
    input  wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] om_req_pos_x,
    input  wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] om_req_pos_y,
    input  om_color_t [NUM_LANES-1:0]       om_req_color,
    input  wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] om_req_depth,
    input  wire [NUM_LANES-1:0]             om_req_face,
    output wire                             om_req_ready,

    output wire [OCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [OCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [OCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [OCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [OCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_om_perf_if perf_om_if();
`ifndef PERF_ENABLE
    assign perf_om_if.mem_reads    = '0;
    assign perf_om_if.mem_writes   = '0;
    assign perf_om_if.mem_latency  = '0;
    assign perf_om_if.stall_cycles = '0;
    `UNUSED_VAR (perf_om_if.mem_reads)
    `UNUSED_VAR (perf_om_if.mem_writes)
    `UNUSED_VAR (perf_om_if.mem_latency)
    `UNUSED_VAR (perf_om_if.stall_cycles)
`endif

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.req_valid     = dcr_write_valid;
    assign dcr_bus_if.req_data.rw   = 1'b1;
    assign dcr_bus_if.req_data.addr = dcr_write_addr;
    assign dcr_bus_if.req_data.data = dcr_write_data;
    `UNUSED_VAR (dcr_bus_if.rsp_valid)
    `UNUSED_VAR (dcr_bus_if.rsp_data)

    VX_om_bus_if #(
        .NUM_LANES (NUM_LANES)
    ) om_bus_if();

    assign om_bus_if.req_valid = om_req_valid;
    assign om_bus_if.req_data.uuid = om_req_uuid;
    assign om_bus_if.req_data.mask = om_req_mask;
    assign om_bus_if.req_data.pos_x = om_req_pos_x;
    assign om_bus_if.req_data.pos_y = om_req_pos_y;
    assign om_bus_if.req_data.color = om_req_color;
    assign om_bus_if.req_data.depth = om_req_depth;
    assign om_bus_if.req_data.face = om_req_face;
    assign om_req_ready = om_bus_if.req_ready;

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) cache_bus_if[OCACHE_NUM_REQS]();

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

    VX_om_core #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES)
    ) om_core (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_om_if   (perf_om_if),
    `endif
        .dcr_bus_if    (dcr_bus_if),
        .om_bus_if    (om_bus_if),
        .cache_bus_if  (cache_bus_if)
    );

endmodule
