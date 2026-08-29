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

// Flat-port wrapper around the full per-core RTU: VX_rtu_unit (SFU shim +
// ray-state register file) wired to VX_rtu_core (traversal engine) over the
// RTU bus. Exposes the SFU execute/result interfaces and the RTCache port as
// flat logic ports for Verilator and FPGA/ASIC synthesis. The SFU execute and
// result payloads are carried as opaque packed buses sized from sfu_execute_t /
// sfu_result_t.

module VX_rtu_top import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_SFU_LANES,
    parameter TAG_WIDTH = RTU_REQ_TAG_WIDTH,
    // RTCache port geometry (line-granular: word size == cache line)
    parameter CACHE_WORD_SIZE  = RTCACHE_WORD_SIZE,
    parameter CACHE_TAG_WIDTH  = RTCACHE_TAG_WIDTH,
    parameter CACHE_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(CACHE_WORD_SIZE),
    parameter EXEC_BITS        = $bits(sfu_execute_t),
    parameter RES_BITS         = $bits(sfu_result_t)
) (
    input  wire clk,
    input  wire reset,

    // -----------------------------------------------------------------------
    // SFU execute interface (slave): one ray-tracing op per dispatch
    // -----------------------------------------------------------------------
    input  wire                      execute_valid,
    input  wire [EXEC_BITS-1:0]      execute_data,
    output wire                      execute_ready,

    // SFU result interface (master)
    output wire                      result_valid,
    output wire [RES_BITS-1:0]       result_data,
    input  wire                      result_ready,

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
    // SFU execute / result interfaces
    // -----------------------------------------------------------------------
    VX_execute_if #(.data_t (sfu_execute_t)) execute_if ();
    assign execute_if.valid = execute_valid;
    assign execute_if.data  = execute_data;
    assign execute_ready    = execute_if.ready;

    VX_result_if #(.data_t (sfu_result_t)) result_if ();
    assign result_valid    = result_if.valid;
    assign result_data     = result_if.data;
    assign result_if.ready = result_ready;

    // -----------------------------------------------------------------------
    // RTU bus (unit master → core slave)
    // -----------------------------------------------------------------------
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) rtu_bus_if ();

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

    assign cache_bus_if.rsp_valid     = cache_rsp_valid;
    assign cache_bus_if.rsp_data.data = cache_rsp_data;
    assign cache_bus_if.rsp_data.tag  = cache_rsp_tag;
    assign cache_rsp_ready            = cache_bus_if.rsp_ready;

    // -----------------------------------------------------------------------
    // DUT: per-core SFU shim + cluster traversal engine
    // -----------------------------------------------------------------------
    VX_rtu_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .CORE_ID     (0),
        .NUM_LANES   (NUM_LANES)
    ) rtu_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (execute_if),
        .result_if  (result_if),
        .rtu_bus_if (rtu_bus_if)
    );

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
