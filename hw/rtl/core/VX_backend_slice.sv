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

// One issue slice carried through issue, execute and commit, serving the warps
// with wid mod ISSUE_WIDTH == slice_id with one block of every execution unit.
// Every slice is the same module: its identity arrives on slice_id/core_id, and
// only the core-wide resources (frontend slices, warp-control resources, memory
// system, SFU buses, WGMMA tile buffers and the metadata AGU) are reached through
// its ports.
module VX_backend_slice import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_TCU_ENABLE
import VX_tcu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING SLICE_STR   = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire [ISSUE_ISW_W-1:0]   slice_id,

`ifdef PERF_ENABLE
    output issue_perf_t     issue_perf,
`endif

    // decoded instructions of the slice's warps
    VX_decode_if.slave      decode_if,

    // issued and retired warps (registered), indexed by wid / ISSUE_WIDTH
    output wire             warp_issued,
    output wire [ISSUE_WIS_W-1:0] warp_issued_wis,
    output wire [PER_ISSUE_WARPS-1:0] committed_wis,

    // scheduler traffic of the slice's ALU and SFU blocks
    VX_branch_ctl_if.master branch_ctl_if,
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_sched_csr_if.slave   sched_csr_if,
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_sched_unlock_if.master sched_unlock_if,
`endif

    // LSU
`ifdef PERF_ENABLE
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rd_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_wr_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rsp_fire,
    output wire [PERF_CTR_BITS-1:0] lsu_coalescer_misses,
`endif
    output wire             lsu_sched_empty,
    output wire             lsu_store_empty,
    output wire             lsu_busy,
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.master    lmem_bus_if [`VX_CFG_NUM_LSU_LANES],
`endif
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_CHANNELS],

    // SFU
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
`endif

    // TCU
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    output wire             tcu_exec_valid,
    output wire             tcu_exec_ready,
    output tcu_execute_t    tcu_exec_data,
    input wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tcu_tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tcu_tbuf_rs2_data,
    input wire              tcu_tbuf_ready,
`endif

    // core-wide warp control and CTA contexts
    VX_warp_shared_if.master warp_shared_if,
    input cta_bcast_t       cta_bcast
);
    `STATIC_ASSERT (BACKEND_SLICES, ("invalid parameter"))

    `SCOPE_IO_SWITCH (2);

    VX_dispatch_if  dispatch_if [NUM_EX_UNITS]();
    VX_commit_if    commit_if [NUM_EX_UNITS]();
    VX_writeback_if writeback_if();

    // issue

    wire issue_fire;
    wire [ISSUE_WIS_W-1:0] issue_wis;

    VX_issue_slice #(
        .INSTANCE_ID (`SFORMATF(("%s-issue%s", INSTANCE_ID, SLICE_STR)))
    ) issue_slice (
        `SCOPE_IO_BIND  (0)
        .clk             (clk),
        .reset           (reset),
        .slice_id        (slice_id),
    `ifdef PERF_ENABLE
        .issue_perf      (issue_perf),
    `endif
        .decode_if       (decode_if),
        .writeback_if    (writeback_if),
        .dispatch_if     (dispatch_if),
        .warp_issued     (issue_fire),
        .warp_issued_wis (issue_wis)
    );

    `BUFFER(warp_issued, issue_fire);
    `BUFFER(warp_issued_wis, issue_wis);

    // execute

    VX_execute_slice #(
        .INSTANCE_ID (`SFORMATF(("%s-execute", INSTANCE_ID))),
        .SLICE_STR   (SLICE_STR)
    ) execute_slice (
        `SCOPE_IO_BIND  (1)
        .clk                  (clk),
        .reset                (reset),
        .core_id              (core_id),
        .slice_id             (slice_id),
        .dispatch_if          (dispatch_if),
        .commit_if            (commit_if),
        .branch_ctl_if        (branch_ctl_if),
    `ifdef PERF_ENABLE
        .lsu_perf_rd_req_fire (lsu_perf_rd_req_fire),
        .lsu_perf_wr_req_fire (lsu_perf_wr_req_fire),
        .lsu_perf_rsp_fire    (lsu_perf_rsp_fire),
        .lsu_coalescer_misses (lsu_coalescer_misses),
    `endif
        .lsu_sched_empty      (lsu_sched_empty),
        .lsu_store_empty      (lsu_store_empty),
        .lsu_busy             (lsu_busy),
    `ifdef VX_CFG_LMEM_ENABLE
        .lmem_bus_if          (lmem_bus_if),
    `endif
        .dcache_bus_if        (dcache_bus_if),
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if       (dxa_req_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if           (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_bus_if           (rtu_bus_if),
        .sched_unlock_if      (sched_unlock_if),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_exec_valid       (tcu_exec_valid),
        .tcu_exec_ready       (tcu_exec_ready),
        .tcu_exec_data        (tcu_exec_data),
        .tcu_tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tcu_tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tcu_tbuf_ready       (tcu_tbuf_ready),
    `endif
        .warp_shared_if       (warp_shared_if),
        .warp_ctl_if          (warp_ctl_if),
        .sched_csr_if         (sched_csr_if),
        .cta_bcast            (cta_bcast)
    );

    // commit

    VX_commit_slice #(
        .INSTANCE_ID (`SFORMATF(("%s-commit", INSTANCE_ID)))
    ) commit_slice (
        .clk           (clk),
        .reset         (reset),
        .commit_if     (commit_if),
        .writeback_if  (writeback_if),
        .committed_wis (committed_wis)
    );

endmodule
