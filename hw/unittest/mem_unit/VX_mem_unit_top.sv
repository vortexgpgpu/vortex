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
    parameter `STRING INSTANCE_ID = ""
) (
    // Clock
    input wire                                          clk,
    input wire                                          reset,

    // LSU block memory clients
    input  wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                   lsu_req_valid,
    input  lsu_req_data_t [`VX_CFG_NUM_LSU_BLOCKS-1:0]         lsu_req_data,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                   lsu_req_ready,

    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                   lsu_rsp_valid,
    output lsu_rsp_data_t [`VX_CFG_NUM_LSU_BLOCKS-1:0]         lsu_rsp_data,
    input  wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                   lsu_rsp_ready,

    // Memory request
    output wire [DCACHE_NUM_REQS-1:0]                   mem_req_valid,
    output wire [DCACHE_NUM_REQS-1:0]                   mem_req_rw,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE-1:0] mem_req_byteen,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [DCACHE_NUM_REQS-1:0][MEM_ATTR_WIDTH-1:0] mem_req_attr,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] mem_req_data,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] mem_req_tag,
    input  wire [DCACHE_NUM_REQS-1:0]                   mem_req_ready,

    // Memory response
    input wire [DCACHE_NUM_REQS-1:0]                    mem_rsp_valid,
    input wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] mem_rsp_data,
    input wire [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] mem_rsp_tag,
    output wire [DCACHE_NUM_REQS-1:0]                   mem_rsp_ready
);
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_bus_if[LSU_NUM_REQS]();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) lsu_dcache_bus_if[DCACHE_NUM_REQS]();

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lsu_mem_paths
        VX_lsu_sched_if client_if();

        assign client_if.req_valid = lsu_req_valid[i];
        assign client_if.req_data  = lsu_req_data[i];
        assign lsu_req_ready[i]    = client_if.req_ready;
        assign lsu_rsp_valid[i]    = client_if.rsp_valid;
        assign lsu_rsp_data[i]     = client_if.rsp_data;
        assign client_if.rsp_ready = lsu_rsp_ready[i];

    `ifdef TCU_META_ENABLE
        VX_lsu_sched_if tcu_mem_if();
        assign tcu_mem_if.req_valid = 1'b0;
        assign tcu_mem_if.req_data  = '0;
        assign tcu_mem_if.rsp_ready = 1'b1;
        `UNUSED_VAR (tcu_mem_if.req_ready)
        `UNUSED_VAR (tcu_mem_if.rsp_valid)
        `UNUSED_VAR (tcu_mem_if.rsp_data)
    `endif

        VX_lsu_mem_path #(
            .INSTANCE_ID (`SFORMATF(("%s-lsu%0d", INSTANCE_ID, i)))
        ) mem_path (
            .clk           (clk),
            .reset         (reset),
        `ifdef PERF_ENABLE
            `UNUSED_PIN (perf_rd_req_fire),
            `UNUSED_PIN (perf_wr_req_fire),
            `UNUSED_PIN (perf_rsp_fire),
            `UNUSED_PIN (coalescer_misses),
        `endif
            .client_if     (client_if),
        `ifdef TCU_META_ENABLE
            .tcu_mem_if    (tcu_mem_if),
        `endif
            `UNUSED_PIN (sched_empty),
            `UNUSED_PIN (store_empty),
            `UNUSED_PIN (busy),
        `ifdef VX_CFG_LMEM_ENABLE
            .lmem_bus_if   (lsu_lmem_bus_if[i * `VX_CFG_NUM_LSU_LANES +: `VX_CFG_NUM_LSU_LANES]),
        `endif
            .dcache_bus_if (lsu_dcache_bus_if[i * DCACHE_CHANNELS +: DCACHE_CHANNELS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) mem_bus_if[DCACHE_NUM_REQS]();

    // memory request
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_mem_req
        assign mem_req_valid[i] = mem_bus_if[i].req_valid;
        assign mem_req_rw[i] = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i] = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i] = mem_bus_if[i].req_data.addr;
        assign mem_req_attr[i] = mem_bus_if[i].req_data.attr;
        assign mem_req_data[i] = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i] = mem_bus_if[i].req_data.tag;
        assign mem_bus_if[i].req_ready = mem_req_ready[i];
    end

    // memory response
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_mem_bus_rsp
        assign mem_bus_if[i].rsp_valid = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.tag = mem_rsp_tag[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

`ifdef PERF_ENABLE
    VX_cache_pkg::cache_perf_t lmem_perf = '0;
`endif

    VX_dcr_flush_if dcr_flush_if();
    assign dcr_flush_if.req = 1'b0;
    `UNUSED_VAR (dcr_flush_if.done)

    VX_mem_unit #(
        .INSTANCE_ID (INSTANCE_ID)
    ) mem_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .lmem_perf     (lmem_perf),
    `endif
    `ifdef VX_CFG_LMEM_ENABLE
        .lmem_bus_if   (lsu_lmem_bus_if),
    `endif
        .lsu_dcache_bus_if (lsu_dcache_bus_if),
        .dcr_flush_if  (dcr_flush_if),
        .flush_l1_only (1'b0),
        .dcache_bus_if (mem_bus_if),
        `UNUSED_PIN    (busy)
    );

endmodule
