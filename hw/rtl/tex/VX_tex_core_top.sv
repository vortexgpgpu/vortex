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

`include "VX_tex_define.vh"

module VX_tex_core_top import VX_gpu_pkg::*; import VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `NUM_THREADS,
    parameter TAG_WIDTH = TEX_REQ_TAG_WIDTH
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    input  wire                             tex_req_valid,
    input  wire [NUM_LANES-1:0]             tex_req_mask,
    input  wire [1:0][NUM_LANES-1:0][31:0]  tex_req_coords,
    input  wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] tex_req_lod,
    input  wire [`VX_TEX_STAGE_BITS-1:0]    tex_req_stage,
    input  wire [TAG_WIDTH-1:0]             tex_req_tag,
    output wire                             tex_req_ready,

    output wire                             tex_rsp_valid,
    output wire [NUM_LANES-1:0][31:0]       tex_rsp_texels,
    output wire [TAG_WIDTH-1:0]             tex_rsp_tag,
    input  wire                             tex_rsp_ready,

    output wire [TCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [TCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [TCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [TCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [TCACHE_NUM_REQS-1:0][TCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [TCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_tex_perf_if perf_tex_if();
`ifndef PERF_ENABLE
    assign perf_tex_if.mem_reads    = '0;
    assign perf_tex_if.mem_latency  = '0;
    assign perf_tex_if.stall_cycles = '0;
    `UNUSED_VAR (perf_tex_if.mem_reads)
    `UNUSED_VAR (perf_tex_if.mem_latency)
    `UNUSED_VAR (perf_tex_if.stall_cycles)
`endif

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.req_valid       = dcr_write_valid;
    assign dcr_bus_if.req_data.rw     = 1'b1;  // unit_top only writes
    assign dcr_bus_if.req_data.addr   = dcr_write_addr;
    assign dcr_bus_if.req_data.data   = dcr_write_data;
    `UNUSED_VAR (dcr_bus_if.rsp_valid)
    `UNUSED_VAR (dcr_bus_if.rsp_data)

    VX_tex_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) tex_bus_if();

    assign tex_bus_if.req_valid = tex_req_valid;
    assign tex_bus_if.req_data.mask = tex_req_mask;
    assign tex_bus_if.req_data.coords = tex_req_coords;
    assign tex_bus_if.req_data.lod = tex_req_lod;
    assign tex_bus_if.req_data.stage = tex_req_stage;
    assign tex_bus_if.req_data.tag = tex_req_tag;
    assign tex_req_ready = tex_bus_if.req_ready;

    assign tex_rsp_valid = tex_bus_if.rsp_valid;
    assign tex_rsp_texels = tex_bus_if.rsp_data.texels;
    assign tex_rsp_tag = tex_bus_if.rsp_data.tag;
    assign tex_bus_if.rsp_ready = tex_rsp_ready;

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) cache_bus_if[TCACHE_NUM_REQS]();

    assign cache_req_valid = cache_bus_if[0].req_valid;
    assign cache_req_rw = cache_bus_if[0].req_data.rw;
    assign cache_req_byteen = cache_bus_if[0].req_data.byteen;
    assign cache_req_addr = cache_bus_if[0].req_data.addr;
    assign cache_req_data = cache_bus_if[0].req_data.data;
    assign cache_req_tag = cache_bus_if[0].req_data.tag;
    assign cache_bus_if[0].req_ready = cache_req_ready;
    `UNUSED_VAR (cache_bus_if[0].req_data.flags)

    assign cache_bus_if[0].rsp_valid = cache_rsp_valid;
    assign cache_bus_if[0].rsp_data.tag = cache_rsp_tag;
    assign cache_bus_if[0].rsp_data.data = cache_rsp_data;
    assign cache_rsp_ready = cache_bus_if[0].rsp_ready;

    VX_tex_core #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH)
    ) tex_core (
        .clk          (clk),
        .reset        (reset),
    `ifdef PERF_ENABLE
        .perf_tex_if  (perf_tex_if),
    `endif
        .dcr_bus_if   (dcr_bus_if),
        .tex_bus_if   (tex_bus_if),
        .cache_bus_if (cache_bus_if)
    );

endmodule
