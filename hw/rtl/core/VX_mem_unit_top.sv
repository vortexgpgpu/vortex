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

module VX_mem_unit_top import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter LSU_WORD_WIDTH = LSU_WORD_SIZE * 8
) (
    // Clock
    input wire                                          clk,
    input wire                                          reset,

    // LSU memory request
    input  wire [`NUM_LSU_BLOCKS-1:0]                   lsu_req_valid,
    input  wire [`NUM_LSU_BLOCKS-1:0]                   lsu_req_rw,
    input  wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0] lsu_req_mask,
    input  wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0][LSU_WORD_SIZE-1:0] lsu_req_byteen,
    input  wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0][LSU_ADDR_WIDTH-1:0] lsu_req_addr,
    input  wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0][`MEM_REQ_FLAGS_WIDTH-1:0] lsu_req_flags,
    input  wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0][LSU_WORD_WIDTH-1:0] lsu_req_data,
    input  wire [`NUM_LSU_BLOCKS-1:0][LSU_TAG_WIDTH-1:0] lsu_req_tag,
    output wire [`NUM_LSU_BLOCKS-1:0]                   lsu_req_ready,

    // LSU memory response
    output wire [`NUM_LSU_BLOCKS-1:0]                   lsu_rsp_valid,
    output wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0] lsu_rsp_mask,
    output wire [`NUM_LSU_BLOCKS-1:0][`NUM_LSU_LANES-1:0][LSU_WORD_WIDTH-1:0] lsu_rsp_data,
    output wire [`NUM_LSU_BLOCKS-1:0][LSU_TAG_WIDTH-1:0] lsu_rsp_tag,
    input  wire [`NUM_LSU_BLOCKS-1:0]                   lsu_rsp_ready,

    // Memory request
    output wire [DCACHE_NUM_REQS-1:0]                   mem_req_valid,
    output wire [DCACHE_NUM_REQS-1:0]                   mem_req_rw,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE-1:0] mem_req_byteen,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [DCACHE_NUM_REQS-1:0][`MEM_REQ_FLAGS_WIDTH-1:0] mem_req_flags,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] mem_req_data,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] mem_req_tag,
    input  wire [DCACHE_NUM_REQS-1:0]                   mem_req_ready,

    // Memory response
    input wire [DCACHE_NUM_REQS-1:0]                    mem_rsp_valid,
    input wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] mem_rsp_data,
    input wire [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] mem_rsp_tag,
    output wire [DCACHE_NUM_REQS-1:0]                   mem_rsp_ready
);
    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_mem_if[`NUM_LSU_BLOCKS]();

    // LSU memory request
    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        assign lsu_mem_if[i].req_valid = lsu_req_valid[i];
        assign lsu_mem_if[i].req_data.rw = lsu_req_rw[i];
        assign lsu_mem_if[i].req_data.mask = lsu_req_mask[i];
        assign lsu_mem_if[i].req_data.byteen = lsu_req_byteen[i];
        assign lsu_mem_if[i].req_data.addr = lsu_req_addr[i];
        assign lsu_mem_if[i].req_data.flags = lsu_req_flags[i];
        assign lsu_mem_if[i].req_data.data = lsu_req_data[i];
        assign lsu_mem_if[i].req_data.tag = lsu_req_tag[i];
        assign lsu_req_ready[i] = lsu_mem_if[i].req_ready;
    end

    // LSU memory response
    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        assign lsu_rsp_valid[i] = lsu_mem_if[i].rsp_valid;
        assign lsu_rsp_mask[i] = lsu_mem_if[i].rsp_data.mask;
        assign lsu_rsp_data[i] = lsu_mem_if[i].rsp_data.data;
        assign lsu_rsp_tag[i] = lsu_mem_if[i].rsp_data.tag;
        assign lsu_mem_if[i].rsp_ready = lsu_rsp_ready[i];
    end

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) mem_bus_if[DCACHE_NUM_REQS]();

    // memory request
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign mem_req_valid[i] = mem_bus_if[i].req_valid;
        assign mem_req_rw[i] = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i] = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i] = mem_bus_if[i].req_data.addr;
        assign mem_req_flags[i] = mem_bus_if[i].req_data.flags;
        assign mem_req_data[i] = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i] = mem_bus_if[i].req_data.tag;
        assign mem_bus_if[i].req_ready = mem_req_ready[i];
    end

    // memory response
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign mem_bus_if[i].rsp_valid = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.tag = mem_rsp_tag[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

`ifdef PERF_ENABLE
    cache_perf_t lmem_perf = '0;
`endif

    VX_mem_unit #(
        .INSTANCE_ID (INSTANCE_ID)
    ) mem_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .lmem_perf     (lmem_perf),
    `endif
        .lsu_mem_if    (lsu_mem_if),
        .dcache_bus_if (mem_bus_if)
    );

endmodule
