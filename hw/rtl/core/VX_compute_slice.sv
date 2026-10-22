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

// A frontend slice and its backend slice: the warps with wid mod ISSUE_WIDTH ==
// slice_id carried through fetch, decode, issue, execute and commit. Used when
// the fetch, issue and every execution-unit width are equal, so frontend slice i
// serves only issue slice i and the blocks of slice i, and the two meet
// directly. Only the core scheduler, the icache, the memory system and the
// shared SFU/TCU services are reached through the ports.
module VX_compute_slice import VX_gpu_pkg::*;
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
    output fetch_perf_t     fetch_perf,
    output issue_perf_t     issue_perf,
`endif

    // core scheduler and icache
    VX_slice_sched_if.slave slice_sched_if,
    VX_mem_bus_if.master    icache_bus_if,
    output wire [PER_ISSUE_WARPS-1:0] committed_wis,

    // core scheduler state
    input wire [PERF_CTR_BITS-1:0] cycles,
    input wire [PERF_CTR_BITS-1:0] instret,
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    input wire              lsu_sched_drained,
    output wire [NCTA_WIDTH-1:0] csr_rd_cta_id,
    input cta_ctx_t         csr_rd_ctx,
`ifdef VX_CFG_VM_ENABLE
    input wire [`VX_CFG_XLEN-1:0] satp,
    output wire             satp_wr_valid,
    output wire [`VX_CFG_XLEN-1:0] satp_wr_data,
`endif

    // LSU
    VX_lsu_sched_if.master  lsu_client_if,

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
`ifdef TCU_META_ENABLE
    output wire             tcu_agu_ld_valid,
    output tcu_execute_t    tcu_agu_ld_data,
    input wire              tcu_agu_ld_ready,
    input wire              tcu_agu_result_valid,
    input tcu_result_t      tcu_agu_result_data,
    output wire             tcu_agu_result_ready,
    input wire              tcu_meta_wr_en,
    input wire [NW_WIDTH-1:0] tcu_meta_wr_wid,
    input wire [4:0]        tcu_meta_wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] tcu_meta_wr_data,
`endif

    // core-wide warp control
    VX_warp_shared_if.master warp_shared_if
);
    `STATIC_ASSERT (COMPUTE_SLICES, ("invalid parameter"))

    localparam SW = PER_FETCH_WARPS;

    `SCOPE_IO_SWITCH (2);

    VX_decode_if        decode_if();
    VX_branch_ctl_if    branch_ctl_if();
    VX_warp_ctl_if      warp_ctl_if();
    VX_sched_csr_if     sched_csr_if();
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_sched_unlock_if  sched_unlock_if();
`endif

    wire warp_issued;
    wire [ISSUE_WIS_W-1:0] warp_issued_wis;

    // slice-local view of the frontend slice's ports: block i serves only frontend slice
    // i, and a warp's fetch-local and issue-local ids are the same
    sched_branch_t [0:0] branch;
    sched_wctl_t [0:0] wctl;
    wire [0:0][NW_WIDTH-1:0] dvstack_wid;
    wire [0:0][DV_STACK_SIZEW-1:0] dvstack_ptr;
    wire [0:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] dvstack_mscratch;
    wire [0:0][NW_WIDTH-1:0] csr_rd_wid;
    sched_csr_rd_t [0:0] csr_rd;
    sched_csr_wr_t [0:0] csr_wr;
`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [0:0] rtu_unlock_valid;
    wire [0:0][NW_WIDTH-1:0] rtu_unlock_wid;
`endif
    wire [SW-1:0] issued;
    wire [SW-1:0] pending_alm_empty;

    VX_frontend_slice #(
        .INSTANCE_ID (INSTANCE_ID),
        .SLICE_STR   ((`VX_CFG_FETCH_WIDTH > 1) ? SLICE_STR : "")
    ) frontend_slice (
        `SCOPE_IO_BIND (0)
        .clk               (clk),
        .reset             (reset),
        .core_id           (core_id),
        .slice_id          (FETCH_FSW_W'(slice_id)),
    `ifdef PERF_ENABLE
        .fetch_perf        (fetch_perf),
    `endif
        .slice_sched_if    (slice_sched_if),
        .icache_bus_if     (icache_bus_if),
        .decode_if         (decode_if),
        .branch            (branch),
        .wctl              (wctl),
        .dvstack_wid       (dvstack_wid),
        .dvstack_ptr       (dvstack_ptr),
        .dvstack_mscratch  (dvstack_mscratch),
        .csr_rd_wid        (csr_rd_wid),
        .csr_rd            (csr_rd),
        .csr_wr            (csr_wr),
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_unlock_valid  (rtu_unlock_valid),
        .rtu_unlock_wid    (rtu_unlock_wid),
    `endif
        .issued            (issued),
        .committed         (committed_wis),
        .pending_alm_empty (pending_alm_empty)
    );

    for (genvar l = 0; l < SW; ++l) begin : g_issued
        assign issued[l] = warp_issued && (warp_issued_wis == ISSUE_WIS_W'(l));
    end

    assign branch[0].valid      = branch_ctl_if.valid;
    assign branch[0].wid        = branch_ctl_if.wid;
    assign branch[0].taken      = branch_ctl_if.taken;
    assign branch[0].dest       = branch_ctl_if.dest;
    assign branch[0].is_trap    = branch_ctl_if.is_trap;
    assign branch[0].is_mret    = branch_ctl_if.is_mret;
    assign branch[0].trap_cause = branch_ctl_if.trap_cause;

    assign wctl[0].tmc_valid    = warp_ctl_if.tmc_valid;
    assign wctl[0].split_valid  = warp_ctl_if.split_valid;
    assign wctl[0].sjoin_valid  = warp_ctl_if.sjoin_valid;
    assign wctl[0].wsync_valid  = warp_ctl_if.wsync_valid;
    assign wctl[0].wid          = warp_ctl_if.wid;
    assign wctl[0].tmc          = warp_ctl_if.tmc;
    assign wctl[0].split        = warp_ctl_if.split;
    assign wctl[0].sjoin        = warp_ctl_if.sjoin;
    assign dvstack_wid[0]       = warp_ctl_if.dvstack_wid;
    assign warp_ctl_if.dvstack_ptr = dvstack_ptr[0];
    assign warp_ctl_if.mscratch    = dvstack_mscratch[0];
    assign warp_ctl_if.warp_pending_alm_empty = pending_alm_empty;
    assign warp_ctl_if.lsu_sched_drained      = lsu_sched_drained;

    assign csr_rd_wid[0]        = sched_csr_if.csr_rd_wid;
    assign csr_rd_cta_id        = sched_csr_if.csr_rd_cta_id;
    assign csr_wr[0].valid      = sched_csr_if.csr_wr_valid;
    assign csr_wr[0].wid        = sched_csr_if.csr_wr_wid;
    assign csr_wr[0].data       = sched_csr_if.csr_wr_data;
    assign csr_wr[0].trap_valid = sched_csr_if.trap_csr_wr_valid;
    assign csr_wr[0].trap_data  = sched_csr_if.trap_csr_wr_data;

    assign sched_csr_if.cycles       = cycles;
    assign sched_csr_if.instret      = instret;
    assign sched_csr_if.active_warps = active_warps;
    assign sched_csr_if.thread_mask  = csr_rd[0].thread_mask;
    assign sched_csr_if.mscratch     = csr_rd[0].mscratch;
    assign sched_csr_if.csr_mstatus  = csr_rd[0].mstatus;
    assign sched_csr_if.csr_mtvec    = csr_rd[0].mtvec;
    assign sched_csr_if.csr_mepc     = csr_rd[0].mepc;
    assign sched_csr_if.csr_mcause   = csr_rd[0].mcause;
    assign sched_csr_if.csr_mtval    = csr_rd[0].mtval;
    assign sched_csr_if.cta_csrs.cta_id       = sched_csr_if.csr_rd_cta_id;
    assign sched_csr_if.cta_csrs.cta_rank     = csr_rd[0].cta_rank;
    assign sched_csr_if.cta_csrs.cta_size     = csr_rd_ctx.cta_size;
    assign sched_csr_if.cta_csrs.block_idx    = csr_rd_ctx.block_idx;
    assign sched_csr_if.cta_csrs.block_dim    = csr_rd_ctx.block_dim;
    assign sched_csr_if.cta_csrs.grid_dim     = csr_rd_ctx.grid_dim;
    assign sched_csr_if.cta_csrs.entry        = csr_rd_ctx.entry;
    assign sched_csr_if.cta_csrs.param        = csr_rd_ctx.param;
    assign sched_csr_if.cta_csrs.lmem_addr    = csr_rd_ctx.lmem_addr;
    assign sched_csr_if.cta_csrs.cluster_size = csr_rd_ctx.cluster_size;
    assign sched_csr_if.cta_lane     = csr_rd[0].cta_lane;
`ifdef VX_CFG_VM_ENABLE
    assign sched_csr_if.csr_satp     = satp;
    assign satp_wr_valid             = sched_csr_if.satp_wr_valid;
    assign satp_wr_data              = sched_csr_if.satp_wr_data;
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    assign rtu_unlock_valid[0] = sched_unlock_if.valid;
    assign rtu_unlock_wid[0]   = sched_unlock_if.wid;
`endif

    VX_backend_slice #(
        .INSTANCE_ID (INSTANCE_ID),
        .SLICE_STR   (SLICE_STR)
    ) backend_slice (
        `SCOPE_IO_BIND (1)
        .clk                  (clk),
        .reset                (reset),
        .core_id              (core_id),
        .slice_id             (slice_id),
    `ifdef PERF_ENABLE
        .issue_perf           (issue_perf),
    `endif
        .decode_if            (decode_if),
        .warp_issued          (warp_issued),
        .warp_issued_wis      (warp_issued_wis),
        .committed_wis        (committed_wis),
        .branch_ctl_if        (branch_ctl_if),
        .warp_ctl_if          (warp_ctl_if),
        .sched_csr_if         (sched_csr_if),
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .sched_unlock_if      (sched_unlock_if),
    `endif
        .lsu_client_if        (lsu_client_if),
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if       (dxa_req_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if           (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_bus_if           (rtu_bus_if),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_exec_valid       (tcu_exec_valid),
        .tcu_exec_ready       (tcu_exec_ready),
        .tcu_exec_data        (tcu_exec_data),
        .tcu_tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tcu_tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tcu_tbuf_ready       (tcu_tbuf_ready),
    `endif
    `ifdef TCU_META_ENABLE
        .tcu_agu_ld_valid     (tcu_agu_ld_valid),
        .tcu_agu_ld_data      (tcu_agu_ld_data),
        .tcu_agu_ld_ready     (tcu_agu_ld_ready),
        .tcu_agu_result_valid (tcu_agu_result_valid),
        .tcu_agu_result_data  (tcu_agu_result_data),
        .tcu_agu_result_ready (tcu_agu_result_ready),
        .tcu_meta_wr_en       (tcu_meta_wr_en),
        .tcu_meta_wr_wid      (tcu_meta_wr_wid),
        .tcu_meta_wr_idx      (tcu_meta_wr_idx),
        .tcu_meta_wr_data     (tcu_meta_wr_data),
    `endif
        .warp_shared_if       (warp_shared_if)
    );

endmodule
