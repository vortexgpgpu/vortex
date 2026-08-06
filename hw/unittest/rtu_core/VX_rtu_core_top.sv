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

// Flat-port wrapper around VX_rtu_core. Exposes the cluster-shared RTU bus and
// the RTCache memory port as individual logic ports for Verilator and
// FPGA/ASIC synthesis (interface modports cannot cross the synthesis top).

module VX_rtu_core_top import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_SFU_LANES,
    parameter TAG_WIDTH = RTU_REQ_TAG_WIDTH,
    // RTCache port geometry (line-granular: word size == cache line)
    parameter CACHE_WORD_SIZE  = RTCACHE_WORD_SIZE,
    parameter CACHE_TAG_WIDTH  = RTCACHE_TAG_WIDTH,
    parameter CACHE_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(CACHE_WORD_SIZE),
    parameter RAY_BITS         = $bits(rtu_ray_t)
) (
    input  wire clk,
    input  wire reset,

    // -----------------------------------------------------------------------
    // RTU request bus (slave): active-lane mask + per-lane ray snapshot
    // -----------------------------------------------------------------------
    input  wire                              rtu_req_valid,
    input  wire [NUM_LANES-1:0]              rtu_req_mask,
    input  wire [NUM_LANES-1:0][RAY_BITS-1:0] rtu_req_rays,
    input  wire [TAG_WIDTH-1:0]              rtu_req_tag,
    output wire                              rtu_req_ready,

    // RTU response bus: per-lane terminal status + closest-hit attributes
    output wire                              rtu_rsp_valid,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_status,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_hit_t,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_hit_u,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_hit_v,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_hit_prim_id,
    output wire [NUM_LANES-1:0][31:0]        rtu_rsp_hit_geometry,
    output wire [TAG_WIDTH-1:0]              rtu_rsp_tag,
    input  wire                              rtu_rsp_ready,

    // -----------------------------------------------------------------------
    // RTCache port (master): node/leaf line fetch
    // -----------------------------------------------------------------------
    output wire                              cache_req_valid,
    output wire                              cache_req_rw,
    output wire [CACHE_WORD_SIZE-1:0]        cache_req_byteen,
    output wire [CACHE_ADDR_WIDTH-1:0]       cache_req_addr,
    output wire [CACHE_WORD_SIZE*8-1:0]      cache_req_data,
    output wire [CACHE_TAG_WIDTH-1:0]        cache_req_tag,
    input  wire                              cache_req_ready,

    input  wire                              cache_rsp_valid,
    input  wire [CACHE_WORD_SIZE*8-1:0]      cache_rsp_data,
    input  wire [CACHE_TAG_WIDTH-1:0]        cache_rsp_tag,
    output wire                              cache_rsp_ready
);
    // -----------------------------------------------------------------------
    // RTU request/response bus
    // -----------------------------------------------------------------------
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) rtu_bus_if ();

    assign rtu_bus_if.req_valid     = rtu_req_valid;
    assign rtu_bus_if.req_data.mask = rtu_req_mask;
    assign rtu_bus_if.req_data.rays = rtu_req_rays;
    assign rtu_bus_if.req_data.tag  = rtu_req_tag;
    assign rtu_req_ready            = rtu_bus_if.req_ready;

    assign rtu_rsp_valid        = rtu_bus_if.rsp_valid;
    assign rtu_rsp_status       = rtu_bus_if.rsp_data.status;
    assign rtu_rsp_hit_t        = rtu_bus_if.rsp_data.hit_t;
    assign rtu_rsp_hit_u        = rtu_bus_if.rsp_data.hit_u;
    assign rtu_rsp_hit_v        = rtu_bus_if.rsp_data.hit_v;
    assign rtu_rsp_hit_prim_id  = rtu_bus_if.rsp_data.hit_prim_id;
    assign rtu_rsp_hit_geometry = rtu_bus_if.rsp_data.hit_geometry;
    assign rtu_rsp_tag          = rtu_bus_if.rsp_data.tag;
    assign rtu_bus_if.rsp_ready = rtu_rsp_ready;

    // -----------------------------------------------------------------------
    // RTCache memory port
    // -----------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (CACHE_WORD_SIZE),
        .TAG_WIDTH (CACHE_TAG_WIDTH)
    ) cache_bus_if ();

    assign cache_req_valid          = cache_bus_if.req_valid;
    assign cache_req_rw             = cache_bus_if.req_data.rw;
    assign cache_req_byteen         = cache_bus_if.req_data.byteen;
    assign cache_req_addr           = cache_bus_if.req_data.addr;
    assign cache_req_data           = cache_bus_if.req_data.data;
    assign cache_req_tag            = cache_bus_if.req_data.tag;
    assign cache_bus_if.req_ready   = cache_req_ready;
    `UNUSED_VAR (cache_bus_if.req_data.attr)

    assign cache_bus_if.rsp_valid    = cache_rsp_valid;
    assign cache_bus_if.rsp_data.data = cache_rsp_data;
    assign cache_bus_if.rsp_data.tag  = cache_rsp_tag;
    assign cache_rsp_ready           = cache_bus_if.rsp_ready;

    // -----------------------------------------------------------------------
    // DUT
    // -----------------------------------------------------------------------
    VX_rtu_core #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH)
    ) rtu_core (
        .clk          (clk),
        .reset        (reset),
        .rtu_bus_if   (rtu_bus_if),
        .cache_bus_if (cache_bus_if)
    );

endmodule
