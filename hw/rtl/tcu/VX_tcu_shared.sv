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

// TCU state shared by every TCU block: the WGMMA tile buffers (a warpgroup's
// B operand is common to its blocks) and the warp-level metadata AGU.
module VX_tcu_shared import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS
) (
`ifdef PERF_ENABLE
    output tcu_perf_t       tcu_perf,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    input wire [BLOCK_SIZE-1:0] exec_valid,
    input wire [BLOCK_SIZE-1:0] exec_ready,
    input tcu_execute_t     exec_data [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data,
    output wire [BLOCK_SIZE-1:0] tbuf_ready_eff,
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master    tcu_lmem_if,
`endif

`ifdef TCU_META_ENABLE
    input wire [BLOCK_SIZE-1:0] agu_ld_valid,
    input tcu_execute_t     agu_ld_data [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0] agu_ld_ready,
    output wire [BLOCK_SIZE-1:0] agu_result_valid,
    output tcu_result_t     agu_result_data [BLOCK_SIZE],
    input wire [BLOCK_SIZE-1:0] agu_result_ready,
    output wire             meta_wr_en,
    output wire [NW_WIDTH-1:0] meta_wr_wid,
    output wire [4:0]       meta_wr_idx,
    output wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] meta_wr_data,
    // TCU_LD memory clients, one per LSU block's scheduler
    VX_lsu_sched_if.master  tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
`endif

    input wire              clk,
    input wire              reset
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifndef VX_CFG_TCU_WGMMA_ENABLE
`ifndef TCU_META_ENABLE
    `UNUSED_PARAM (BLOCK_SIZE)
    `UNUSED_VAR ({clk, reset})
`endif
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    VX_tcu_wgmma #(
        .INSTANCE_ID (`SFORMATF(("%s-wgmma", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) wgmma (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
        .exec_valid     (exec_valid),
        .exec_ready     (exec_ready),
        .exec_data      (exec_data),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready_eff (tbuf_ready_eff)
    );
`else
`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = '0;
    assign tcu_perf.tbuf_cache_hits = '0;
    assign tcu_perf.lmem_reads      = '0;
    assign tcu_perf.wgmma_instrs    = '0;
    assign tcu_perf.wgmma_stalls    = '0;
`endif
`endif

`ifdef TCU_META_ENABLE
    // The AGU streams NUM_TCU_LANES operands over the LSU memory client, whose
    // request/response mask and data are sized by NUM_LSU_LANES; the two lane
    // counts must match for the per-lane completion tracking to be correct.
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_LANES == `VX_CFG_NUM_LSU_LANES, ("VX_tcu_agu requires NUM_TCU_LANES == NUM_LSU_LANES"));

    // With one LSU block per TCU block, each block loads its metadata
    // through the scheduler of the LSU block that serves the same warps;
    // otherwise every block shares LSU block 0's scheduler.
    localparam AGU_PORTS = (`VX_CFG_NUM_LSU_BLOCKS == BLOCK_SIZE) ? BLOCK_SIZE : 1;
    for (genvar b = AGU_PORTS; b < `VX_CFG_NUM_LSU_BLOCKS; ++b) begin : g_mem_tieoff
        assign tcu_mem_if[b].req_valid = 1'b0;
        assign tcu_mem_if[b].req_data  = '0;
        assign tcu_mem_if[b].rsp_ready = 1'b1;
        `UNUSED_VAR (tcu_mem_if[b].req_ready)
        `UNUSED_VAR (tcu_mem_if[b].rsp_valid)
        `UNUSED_VAR (tcu_mem_if[b].rsp_data)
    end

    VX_lsu_sched_if agu_mem_if [AGU_PORTS]();
    for (genvar b = 0; b < AGU_PORTS; ++b) begin : g_mem_port
        assign tcu_mem_if[b].req_valid = agu_mem_if[b].req_valid;
        assign tcu_mem_if[b].req_data  = agu_mem_if[b].req_data;
        assign agu_mem_if[b].req_ready = tcu_mem_if[b].req_ready;
        assign agu_mem_if[b].rsp_valid = tcu_mem_if[b].rsp_valid;
        assign agu_mem_if[b].rsp_data  = tcu_mem_if[b].rsp_data;
        assign tcu_mem_if[b].rsp_ready = agu_mem_if[b].rsp_ready;
    end

    VX_tcu_agu #(
        .INSTANCE_ID (`SFORMATF(("%s-agu", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE),
        .NUM_LANES   (`VX_CFG_NUM_TCU_LANES),
        .NUM_PORTS   (AGU_PORTS)
    ) agu (
        .clk                (clk),
        .reset              (reset),
        .per_block_ld_valid (agu_ld_valid),
        .per_block_ld_data  (agu_ld_data),
        .per_block_ld_ready (agu_ld_ready),
        .client_if          (agu_mem_if),
        .meta_wr_en         (meta_wr_en),
        .meta_wr_wid        (meta_wr_wid),
        .meta_wr_idx        (meta_wr_idx),
        .meta_wr_data       (meta_wr_data),
        .result_valid       (agu_result_valid),
        .result_data        (agu_result_data),
        .result_ready       (agu_result_ready)
    );
`endif

endmodule
