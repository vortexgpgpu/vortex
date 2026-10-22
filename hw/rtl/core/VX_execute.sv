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

module VX_execute import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_TCU_ENABLE
import VX_tcu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

    // LSU memory paths
`ifdef PERF_ENABLE
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rd_req_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_wr_req_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rsp_fire,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] lsu_coalescer_misses,
`endif
`ifdef TCU_META_ENABLE
    VX_lsu_sched_if.slave   lsu_tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
`endif
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_sched_empty,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_store_empty,
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.master    lmem_bus_if [LSU_NUM_REQS],
`endif
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    // dispatch interface
    VX_dispatch_if.slave    dispatch_if [NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH],

    // commit interface
    VX_commit_if.master     commit_if [NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH],

    // per-block ports of the core-wide SFU and TCU services
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // FWD raster payload-stage write port (from the core-level distributor).
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_sched_unlock_if.master sched_unlock_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_exec_valid,
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_exec_ready,
    output tcu_execute_t    tcu_exec_data [`VX_CFG_NUM_TCU_BLOCKS],
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tcu_tbuf_rs1_data,
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tcu_tbuf_rs2_data,
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_tbuf_ready,
`endif
`ifdef TCU_META_ENABLE
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_ld_valid,
    output tcu_execute_t    tcu_agu_ld_data [`VX_CFG_NUM_TCU_BLOCKS],
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_ld_ready,
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_result_valid,
    input tcu_result_t      tcu_agu_result_data [`VX_CFG_NUM_TCU_BLOCKS],
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_result_ready,
    input wire              tcu_meta_wr_en,
    input wire [NW_WIDTH-1:0] tcu_meta_wr_wid,
    input wire [4:0]        tcu_meta_wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] tcu_meta_wr_data,
`endif

    // scheduler interfaces
    VX_sched_csr_if.slave   sched_csr_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_branch_ctl_if.master branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS],
    VX_warp_shared_if.master warp_shared_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_warp_ctl_if.master   warp_ctl_if [`VX_CFG_NUM_SFU_BLOCKS]
);

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if fpu_csr_if[`VX_CFG_NUM_FPU_BLOCKS]();
`endif

    VX_alu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-alu", INSTANCE_ID)))
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_ALU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_ALU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .branch_ctl_if  (branch_ctl_if)
    );

`ifdef SCOPE
    localparam SCOPE_COUNT = 1 + `VX_CFG_EXT_TCU_ENABLED;
`endif
    `SCOPE_IO_SWITCH (SCOPE_COUNT);

    VX_lsu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-lsu", INSTANCE_ID)))
    ) lsu_unit (
        `SCOPE_IO_BIND  (0)
        .clk             (clk),
        .reset           (reset),
        .core_id         (core_id),
        .dispatch_if         (dispatch_if[EX_LSU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if           (commit_if[EX_LSU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
    `ifdef PERF_ENABLE
        .perf_rd_req_fire    (lsu_perf_rd_req_fire),
        .perf_wr_req_fire    (lsu_perf_wr_req_fire),
        .perf_rsp_fire       (lsu_perf_rsp_fire),
        .coalescer_misses    (lsu_coalescer_misses),
    `endif
    `ifdef TCU_META_ENABLE
        .tcu_mem_if          (lsu_tcu_mem_if),
    `endif
        .sched_empty         (lsu_sched_empty),
        .store_empty         (lsu_store_empty),
    `ifdef VX_CFG_LMEM_ENABLE
        .lmem_bus_if         (lmem_bus_if),
    `endif
        .dcache_bus_if       (dcache_bus_if)
    );

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-fpu", INSTANCE_ID)))
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_FPU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_FPU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .fpu_csr_if     (fpu_csr_if)
    );
`endif

`ifdef VX_CFG_EXT_TCU_ENABLE
    VX_tcu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tcu", INSTANCE_ID)))
    ) tcu_unit (
        `SCOPE_IO_BIND  (1)
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
        .agu_ld_valid     (tcu_agu_ld_valid),
        .agu_ld_data      (tcu_agu_ld_data),
        .agu_ld_ready     (tcu_agu_ld_ready),
        .agu_result_valid (tcu_agu_result_valid),
        .agu_result_data  (tcu_agu_result_data),
        .agu_result_ready (tcu_agu_result_ready),
        .meta_wr_en       (tcu_meta_wr_en),
        .meta_wr_wid      (tcu_meta_wr_wid),
        .meta_wr_idx      (tcu_meta_wr_idx),
        .meta_wr_data     (tcu_meta_wr_data),
    `endif
        .dispatch_if      (dispatch_if[EX_TCU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if        (commit_if[EX_TCU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH])
    );
`endif

    VX_sfu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-sfu", INSTANCE_ID)))
    ) sfu_unit (
        .clk            (clk),
        .reset          (reset),
        .core_id        (core_id),
        .dispatch_if    (dispatch_if[EX_SFU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_SFU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
    `ifdef VX_CFG_EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if (dxa_req_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if     (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_bus_if     (rtu_bus_if),
        .sched_unlock_if (sched_unlock_if),
    `endif
        .sched_csr_if   (sched_csr_if),
        .warp_shared_if (warp_shared_if),
        .warp_ctl_if    (warp_ctl_if)
    );

endmodule
