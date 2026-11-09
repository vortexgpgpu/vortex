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
// B operand is common to its blocks) and the tensor memory.
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

`ifdef VX_CFG_TCU_TMEM_ENABLE
    VX_tcu_job_if.tmem      job_if [BLOCK_SIZE],
    // tc.commit arrivals
    VX_tcu_bar_if.master    tcu_bar_if,
`endif

    input wire              clk,
    input wire              reset
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifndef VX_CFG_TCU_WGMMA_ENABLE
    `UNUSED_PARAM (BLOCK_SIZE)
    `UNUSED_VAR ({clk, reset})
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    wire [BLOCK_SIZE-1:0] tj_op_valid, tj_op_ready;
    tcu_execute_t         tj_op_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] tj_rsp_valid, tj_rsp_ready;
    tcu_result_t          tj_rsp_data [BLOCK_SIZE];
    wire                  job_valid, job_fire;
    tcu_execute_t         job_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] job_core_ready;
    wire [BLOCK_SIZE-1:0] jres_valid;
    wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] jres_data;
    tcu_tbuf_req_t [BLOCK_SIZE-1:0] job_tbuf_req;
    wire                  job_a_fill_valid;
    wire [3:0]            job_a_fill_k;
    wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] job_a_fill_data;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta, job_meta_hi;
`endif
    VX_mem_bus_if #(
        .DATA_SIZE  (`VX_CFG_LMEM_NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH  (TCU_LMEM_BLK_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (`VX_CFG_LMEM_LOG_SIZE - $clog2(`VX_CFG_XLEN / 8) - $clog2(`VX_CFG_LMEM_NUM_BANKS))
    ) job_cp_lmem_if();

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_job_if
        assign tj_op_valid[b]       = job_if[b].op_valid;
        assign tj_op_data[b]        = job_if[b].op_data;
        assign job_if[b].op_ready   = tj_op_ready[b];
        assign job_if[b].rsp_valid  = tj_rsp_valid[b];
        assign job_if[b].rsp_data   = tj_rsp_data[b];
        assign tj_rsp_ready[b]      = job_if[b].rsp_ready;
        assign job_if[b].job_valid  = job_valid;
        assign job_if[b].job_fire   = job_fire;
        assign job_if[b].job_data   = job_data[b];
        assign job_core_ready[b]    = job_if[b].core_ready;
        assign jres_valid[b]        = job_if[b].jres_valid;
        assign jres_data[b]         = job_if[b].jres_data;
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        assign job_if[b].job_meta    = job_meta[b];
        assign job_if[b].job_meta_hi = job_meta_hi[b];
    `endif
    end

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] tj_perf_busy, tj_perf_opnd, tj_perf_other, tj_perf_idle, tj_perf_xfers;
`endif

    VX_tcu_tmem #(
        .INSTANCE_ID (`SFORMATF(("%s-tmem", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) tmem (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .perf_busy      (tj_perf_busy),
        .perf_opnd      (tj_perf_opnd),
        .perf_other     (tj_perf_other),
        .perf_idle      (tj_perf_idle),
        .perf_xfers     (tj_perf_xfers),
    `endif
        .op_valid       (tj_op_valid),
        .op_data        (tj_op_data),
        .op_ready       (tj_op_ready),
        .rsp_valid      (tj_rsp_valid),
        .rsp_data       (tj_rsp_data),
        .rsp_ready      (tj_rsp_ready),
        .job_valid      (job_valid),
        .job_data       (job_data),
        .core_ready     (job_core_ready),
        .job_fire       (job_fire),
        .tbuf_req       (job_tbuf_req),
        .a_fill_valid   (job_a_fill_valid),
        .a_fill_k       (job_a_fill_k),
        .a_fill_data    (job_a_fill_data),
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        .job_meta       (job_meta),
        .job_meta_hi    (job_meta_hi),
    `endif
        .jres_valid     (jres_valid),
        .jres_data      (jres_data),
        .tcu_bar_if     (tcu_bar_if),
        .cp_lmem_if     (job_cp_lmem_if)
    );
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
`ifdef PERF_ENABLE
    tcu_perf_t wgmma_perf;
    always_comb begin
        tcu_perf = wgmma_perf;
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        tcu_perf.mma_busy   = tj_perf_busy;
        tcu_perf.opnd_waits = tj_perf_opnd;
        tcu_perf.job_other  = tj_perf_other;
        tcu_perf.idle       = tj_perf_idle;
        tcu_perf.tmem_xfers = tj_perf_xfers;
    `endif
    end
`endif

    VX_tcu_wgmma #(
        .INSTANCE_ID (`SFORMATF(("%s-wgmma", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) wgmma (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (wgmma_perf),
    `endif
        .exec_valid     (exec_valid),
        .exec_ready     (exec_ready),
        .exec_data      (exec_data),
        .tcu_lmem_if    (tcu_lmem_if),
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        .job_req        (job_tbuf_req),
        .job_a_fill_valid (job_a_fill_valid),
        .job_a_fill_k   (job_a_fill_k),
        .job_a_fill_data (job_a_fill_data),
        .job_cp_lmem_if (job_cp_lmem_if),
    `endif
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
    assign tcu_perf.mma_busy        = '0;
    assign tcu_perf.opnd_waits      = '0;
    assign tcu_perf.job_other       = '0;
    assign tcu_perf.idle            = '0;
    assign tcu_perf.tmem_xfers      = '0;
`endif
`endif

endmodule
