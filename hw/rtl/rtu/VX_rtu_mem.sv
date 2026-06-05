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

// VX_rtu_mem — read-only node/leaf fetch engine. A CW-BVH4 node is exactly
// one cache line, so each request fetches one aligned 64 B line through the
// RTCache port. Converts a byte address to a line address, forwards a
// scheduler tag through the request, and returns the line with its tag.
// Outstanding requests are bounded by the cache's own MSHR; the scheduler
// tag distinguishes responses (one in flight per context in Phase 1).

`include "VX_define.vh"

module VX_rtu_mem import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter TAG_WIDTH = 1,
    parameter LINE_SIZE = `VX_CFG_MEM_BLOCK_SIZE
) (
    input  wire clk,
    input  wire reset,

    // scheduler-side request / response
    input  wire                              req_valid,
    input  wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] req_addr,   // byte address
    input  wire [TAG_WIDTH-1:0]              req_tag,
    output wire                              req_ready,

    output wire                              rsp_valid,
    output wire [LINE_SIZE*8-1:0]            rsp_data,    // fetched line
    output wire [TAG_WIDTH-1:0]              rsp_tag,
    input  wire                              rsp_ready,

    // RTCache port
    VX_mem_bus_if.master                     cache_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LINE_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE);

    // request: read the aligned line containing req_addr
    assign cache_bus_if.req_valid        = req_valid;
    assign cache_bus_if.req_data.rw      = 1'b0;
    assign cache_bus_if.req_data.addr    = req_addr[`VX_CFG_MEM_ADDR_WIDTH-1 -: LINE_ADDR_W];
    assign cache_bus_if.req_data.data    = '0;
    assign cache_bus_if.req_data.byteen  = {LINE_SIZE{1'b1}};
    assign cache_bus_if.req_data.tag.uuid  = '0;
    assign cache_bus_if.req_data.tag.value = req_tag;
    assign cache_bus_if.req_data.attr = '0;
    assign req_ready = cache_bus_if.req_ready;

    // response: deliver the fetched line + its scheduler tag
    assign rsp_valid              = cache_bus_if.rsp_valid;
    assign rsp_data               = cache_bus_if.rsp_data.data;
    assign rsp_tag                = cache_bus_if.rsp_data.tag.value;
    assign cache_bus_if.rsp_ready = rsp_ready;
    `UNUSED_VAR (cache_bus_if.rsp_data.tag.uuid)

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (req_valid && req_ready) begin
            `TRACE(2, ("%t: %s fetch: addr=0x%0h, tag=0x%0h\n",
                $time, INSTANCE_ID, req_addr, req_tag))
        end
    end
`endif

endmodule
