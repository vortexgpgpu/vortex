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

// Synthesis wrapper of the tensor-core unit at any issue width: VX_tcu_unit
// plus VX_tcu_shared (tile buffers and, with VX_CFG_TCU_TMEM_ENABLE, tensor
// memory), every dispatch slot, commit slot and memory port at a top-level
// port so nothing is optimized away.
module VX_tcu_wide_top import VX_gpu_pkg::*, VX_tcu_pkg::*; (
    `SCOPE_IO_DECL

    input  wire clk,
    input  wire reset,

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    output wire                                             tcu_lmem_req_valid,
    input  wire                                             tcu_lmem_req_ready,
    output wire [`VX_CFG_LMEM_LOG_SIZE-`CLOG2(LSU_WORD_SIZE)-`CLOG2(`VX_CFG_LMEM_NUM_BANKS)-1:0] tcu_lmem_req_addr,
    input  wire                                             tcu_lmem_rsp_valid,
    input  wire [`VX_CFG_LMEM_NUM_BANKS*`VX_CFG_XLEN-1:0]   tcu_lmem_rsp_data,
`endif

`ifdef TCU_META_ENABLE
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                              tcu_mem_req_valid,
    input  wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                              tcu_mem_req_ready,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][$bits(lsu_req_data_t)-1:0]   tcu_mem_req_data,
    input  wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                              tcu_mem_rsp_valid,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]                              tcu_mem_rsp_ready,
    input  wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][$bits(lsu_rsp_data_t)-1:0]   tcu_mem_rsp_data,
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    output wire                                tcu_bar_valid,
    output wire [$bits(tcu_bar_t)-1:0]         tcu_bar_data,
    input  wire                                tcu_bar_ready,
    input  wire                                cta_exit_valid,
    input  wire [NCTA_WIDTH-1:0]               cta_exit_id,
`endif

    input  wire [`VX_CFG_ISSUE_WIDTH-1:0]                         dispatch_valid,
    output wire [`VX_CFG_ISSUE_WIDTH-1:0]                         dispatch_ready,
    input  wire [`VX_CFG_ISSUE_WIDTH-1:0][$bits(dispatch_t)-1:0]  dispatch_data,

    output wire [`VX_CFG_ISSUE_WIDTH-1:0]                         commit_valid,
    input  wire [`VX_CFG_ISSUE_WIDTH-1:0]                         commit_ready,
    output wire [`VX_CFG_ISSUE_WIDTH-1:0][$bits(commit_t)-1:0]    commit_data
);
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    localparam TCU_LMEM_BANK_ADDR_W = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE) - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);

    VX_mem_bus_if #(
        .DATA_SIZE  (`VX_CFG_LMEM_NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH  (UUID_WIDTH),
        .ADDR_WIDTH (TCU_LMEM_BANK_ADDR_W)
    ) tcu_lmem_if();

    assign tcu_lmem_req_valid         = tcu_lmem_if.req_valid;
    assign tcu_lmem_if.req_ready      = tcu_lmem_req_ready;
    assign tcu_lmem_req_addr          = tcu_lmem_if.req_data.addr;
    assign tcu_lmem_if.rsp_valid      = tcu_lmem_rsp_valid;
    assign tcu_lmem_if.rsp_data.data  = tcu_lmem_rsp_data;
    assign tcu_lmem_if.rsp_data.tag   = '0;
    `UNUSED_VAR (tcu_lmem_if.req_data)
    `UNUSED_VAR (tcu_lmem_if.rsp_ready)
`endif

`ifdef TCU_META_ENABLE
    VX_lsu_sched_if tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS]();
    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_mem
        assign tcu_mem_req_valid[i]    = tcu_mem_if[i].req_valid;
        assign tcu_mem_req_data[i]     = tcu_mem_if[i].req_data;
        assign tcu_mem_if[i].req_ready = tcu_mem_req_ready[i];
        assign tcu_mem_if[i].rsp_valid = tcu_mem_rsp_valid[i];
        assign tcu_mem_if[i].rsp_data  = tcu_mem_rsp_data[i];
        assign tcu_mem_rsp_ready[i]    = tcu_mem_if[i].rsp_ready;
    end
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    VX_tcu_job_if job_if [`VX_CFG_NUM_TCU_BLOCKS]();
    VX_tcu_bar_if tcu_bar_if();
    assign tcu_bar_valid        = tcu_bar_if.req_valid;
    assign tcu_bar_data         = tcu_bar_if.req_data;
    assign tcu_bar_if.req_ready = tcu_bar_ready;
`endif

    VX_dispatch_if dispatch_if[`VX_CFG_ISSUE_WIDTH]();
    VX_commit_if   commit_if  [`VX_CFG_ISSUE_WIDTH]();

    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_slots
        assign dispatch_if[i].valid = dispatch_valid[i];
        assign dispatch_if[i].data  = dispatch_data[i];
        assign dispatch_ready[i]    = dispatch_if[i].ready;
        assign commit_valid[i]      = commit_if[i].valid;
        assign commit_data[i]       = commit_if[i].data;
        assign commit_if[i].ready   = commit_ready[i];
    end

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_exec_valid, tcu_exec_ready;
    tcu_execute_t                     tcu_exec_data [`VX_CFG_NUM_TCU_BLOCKS];
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tcu_tbuf_rs1_data;
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tcu_tbuf_rs2_data;
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_tbuf_ready;
`endif

    VX_tcu_unit #(
        .INSTANCE_ID ("tcu_wide_top")
    ) tcu_unit (
        `SCOPE_IO_BIND (0)
        .clk              (clk),
        .reset            (reset),
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .exec_valid       (tcu_exec_valid),
        .exec_ready       (tcu_exec_ready),
        .exec_data        (tcu_exec_data),
        .tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tbuf_ready       (tcu_tbuf_ready),
    `endif
    `ifdef TCU_META_ENABLE
        .tcu_mem_if       (tcu_mem_if),
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        .job_if           (job_if),
    `endif
        .dispatch_if      (dispatch_if),
        .commit_if        (commit_if)
    );

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    VX_tcu_shared #(
        .INSTANCE_ID ("tcu_wide_top"),
        .BLOCK_SIZE  (`VX_CFG_NUM_TCU_BLOCKS)
    ) tcu_shared (
        .exec_valid       (tcu_exec_valid),
        .exec_ready       (tcu_exec_ready),
        .exec_data        (tcu_exec_data),
        .tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tbuf_ready_eff   (tcu_tbuf_ready),
        .tcu_lmem_if      (tcu_lmem_if),
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        .job_if           (job_if),
        .tcu_bar_if       (tcu_bar_if),
        .cta_exit_valid   (cta_exit_valid),
        .cta_exit_id      (cta_exit_id),
    `endif
        .clk              (clk),
        .reset            (reset)
    );
`endif

endmodule
