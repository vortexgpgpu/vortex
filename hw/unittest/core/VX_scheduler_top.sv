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

// Standalone synthesis wrapper for VX_scheduler: flattens the interface
// boundary into ports so the scheduler's own area/timing can be measured per
// divergence architecture (VX_CFG_DIVERGE_TYPE).
module VX_scheduler_top import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire                              clk,
    input wire                              reset,

    // warp control
    input wire                              wctl_wspawn_valid,
    input wire                              wctl_tmc_valid,
    input wire                              wctl_split_valid,
    input wire                              wctl_sjoin_valid,
    input wire                              wctl_bar_valid,
    input wire                              wctl_wsync_valid,
`ifdef VX_CFG_DIVERGE_TYPE_SCS
    input wire                              wctl_yield_valid,
    input wire                              wctl_pred_park_valid,
    input wire [`VX_CFG_NUM_THREADS-1:0]    wctl_pred_park_tmask,
    input wire                              wctl_pred_restore_valid,
`endif
`ifdef VX_CFG_DIVERGE_TYPE_ITS
    input its_bar_t                         wctl_its,
    input wire [PC_BITS-1:0]                wctl_its_pc,
`endif
    input wire [NW_WIDTH-1:0]               wctl_wid,
    input wspawn_t                          wctl_wspawn,
    input tmc_t                             wctl_tmc,
    input split_t                           wctl_split,
    input join_t                            wctl_sjoin,
    input barrier_t                         wctl_bar,
    input wire [BAR_ADDR_W-1:0]             wctl_bar_addr,
    output wire                             wctl_bar_phase,
    output wire [`VX_CFG_NUM_WARPS-1:0]     wctl_warp_pending_alm_empty,
    output wire                             wctl_lsu_sched_drained,
    input wire [NW_WIDTH-1:0]               wctl_dvstack_wid,
    output wire [DV_STACK_SIZEW-1:0]        wctl_dvstack_ptr,

    // branch resolution (per ALU block)
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]                 branch_valid,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][NW_WIDTH-1:0]   branch_wid,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]                 branch_taken,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][PC_BITS-1:0]    branch_dest,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]                 branch_is_trap,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]                 branch_is_mret,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][3:0]            branch_trap_cause,
`ifdef VX_CFG_DIVERGE_TYPE_ITS
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0] branch_taken_mask,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0] branch_tmask,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] branch_dest_its,
    input wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][PC_BITS-1:0]    branch_ntaken_pc,
`endif

    // decode / issue / commit feedback
    input wire                              decode_valid,
    input wire                              decode_unlock,
`ifdef VX_CFG_EXT_C_ENABLE
    input wire                              decode_is_rvc,
`endif
    input wire [NW_WIDTH-1:0]               decode_wid,
    input wire [`VX_CFG_ISSUE_WIDTH-1:0]    issue_valid,
    input wire [`VX_CFG_ISSUE_WIDTH-1:0][ISSUE_WIS_W-1:0] issue_wis,
    input wire [`VX_CFG_NUM_WARPS-1:0]      committed_warps,

    // KMU bus
    input wire                              kmu_valid,
    input wire                              kmu_kind,
    input wire                              kmu_eop,
    input wire [KMU_DEST_W-1:0]             kmu_dest,
    input wire [KMU_DATAW-1:0]              kmu_data,
    output wire                             kmu_ready,

    // schedule output
    output wire                             schedule_valid,
    output schedule_t                       schedule_data,
    input wire                              schedule_ready,
    input wire [`VX_CFG_NUM_WARPS-1:0]      schedule_ibuf_pop,

    // CSR side
    output wire [PERF_CTR_BITS-1:0]         csr_cycles,
    output wire [PERF_CTR_BITS-1:0]         csr_instret,
    output wire [`VX_CFG_NUM_WARPS-1:0]     csr_active_warps,
    output wire [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] csr_thread_masks,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] csr_mscratch,
    output cta_csrs_t                       csr_cta_csrs,
    output cta_lane_t [`VX_CFG_NUM_THREADS-1:0] csr_cta_lane,
    output wire [`VX_CFG_XLEN-1:0]          csr_mstatus,
    output wire [`VX_CFG_XLEN-1:0]          csr_mtvec,
    output wire [`VX_CFG_XLEN-1:0]          csr_mepc,
    output wire [`VX_CFG_XLEN-1:0]          csr_mcause,
    output wire [`VX_CFG_XLEN-1:0]          csr_mtval,
    input wire [NW_WIDTH-1:0]               csr_rd_wid,
    input wire [NCTA_WIDTH-1:0]             csr_rd_cta_id,
    input wire                              csr_wr_valid,
    input wire [NW_WIDTH-1:0]               csr_wr_wid,
    input wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] csr_wr_data,
    input wire                              trap_csr_wr_valid,
    input wire [`VX_CSR_ADDR_BITS-1:0]      trap_csr_wr_addr,
    input wire [`VX_CFG_XLEN-1:0]           trap_csr_wr_data,

    // global barrier bus
    output wire                             gbar_req_valid,
    output gbar_req_t                       gbar_req_data,
    input wire                              gbar_req_ready,
    input wire                              gbar_rsp_valid,
    input gbar_rsp_t                        gbar_rsp_data,
    output wire                             gbar_rsp_ready,

    // status
    output wire                             busy
);
    VX_warp_ctl_if warp_ctl_if();
    assign warp_ctl_if.wspawn_valid = wctl_wspawn_valid;
    assign warp_ctl_if.tmc_valid    = wctl_tmc_valid;
    assign warp_ctl_if.split_valid  = wctl_split_valid;
    assign warp_ctl_if.sjoin_valid  = wctl_sjoin_valid;
    assign warp_ctl_if.bar_valid    = wctl_bar_valid;
    assign warp_ctl_if.wsync_valid  = wctl_wsync_valid;
`ifdef VX_CFG_DIVERGE_TYPE_SCS
    assign warp_ctl_if.yield_valid        = wctl_yield_valid;
    assign warp_ctl_if.pred_park_valid    = wctl_pred_park_valid;
    assign warp_ctl_if.pred_park_tmask    = wctl_pred_park_tmask;
    assign warp_ctl_if.pred_restore_valid = wctl_pred_restore_valid;
`endif
`ifdef VX_CFG_DIVERGE_TYPE_ITS
    assign warp_ctl_if.its    = wctl_its;
    assign warp_ctl_if.its_pc = wctl_its_pc;
`endif
    assign warp_ctl_if.wid      = wctl_wid;
    assign warp_ctl_if.wspawn   = wctl_wspawn;
    assign warp_ctl_if.tmc      = wctl_tmc;
    assign warp_ctl_if.split    = wctl_split;
    assign warp_ctl_if.sjoin    = wctl_sjoin;
    assign warp_ctl_if.bar      = wctl_bar;
    assign warp_ctl_if.bar_addr = wctl_bar_addr;
    assign wctl_bar_phase              = warp_ctl_if.bar_phase;
    assign wctl_warp_pending_alm_empty = warp_ctl_if.warp_pending_alm_empty;
    assign wctl_lsu_sched_drained      = warp_ctl_if.lsu_sched_drained;
    assign warp_ctl_if.dvstack_wid = wctl_dvstack_wid;
    assign wctl_dvstack_ptr        = warp_ctl_if.dvstack_ptr;

    VX_branch_ctl_if branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS]();
    for (genvar i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin : g_branch_ctl
        assign branch_ctl_if[i].valid      = branch_valid[i];
        assign branch_ctl_if[i].wid        = branch_wid[i];
        assign branch_ctl_if[i].taken      = branch_taken[i];
        assign branch_ctl_if[i].dest       = branch_dest[i];
        assign branch_ctl_if[i].is_trap    = branch_is_trap[i];
        assign branch_ctl_if[i].is_mret    = branch_is_mret[i];
        assign branch_ctl_if[i].trap_cause = branch_trap_cause[i];
    `ifdef VX_CFG_DIVERGE_TYPE_ITS
        assign branch_ctl_if[i].taken_mask = branch_taken_mask[i];
        assign branch_ctl_if[i].tmask      = branch_tmask[i];
        assign branch_ctl_if[i].dest_its   = branch_dest_its[i];
        assign branch_ctl_if[i].ntaken_pc  = branch_ntaken_pc[i];
    `endif
    end

    VX_decode_sched_if decode_sched_if();
    assign decode_sched_if.valid  = decode_valid;
    assign decode_sched_if.unlock = decode_unlock;
`ifdef VX_CFG_EXT_C_ENABLE
    assign decode_sched_if.is_rvc = decode_is_rvc;
`endif
    assign decode_sched_if.wid    = decode_wid;

    VX_issue_sched_if issue_sched_if [`VX_CFG_ISSUE_WIDTH]();
    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_issue_sched
        assign issue_sched_if[i].valid = issue_valid[i];
        assign issue_sched_if[i].wis   = issue_wis[i];
    end

    VX_commit_sched_if commit_sched_if();
    assign commit_sched_if.committed_warps = committed_warps;

    VX_kmu_bus_if kmu_bus_if();
    assign kmu_bus_if.valid = kmu_valid;
    assign kmu_bus_if.kind  = kmu_kind;
    assign kmu_bus_if.eop   = kmu_eop;
    assign kmu_bus_if.dest  = kmu_dest;
    assign kmu_bus_if.data  = kmu_data;
    assign kmu_ready = kmu_bus_if.ready;

    VX_schedule_if schedule_if();
    assign schedule_valid = schedule_if.valid;
    assign schedule_data  = schedule_if.data;
    assign schedule_if.ready    = schedule_ready;
    assign schedule_if.ibuf_pop = schedule_ibuf_pop;

    VX_sched_csr_if sched_csr_if();
    assign csr_cycles       = sched_csr_if.cycles;
    assign csr_instret      = sched_csr_if.instret;
    assign csr_active_warps = sched_csr_if.active_warps;
    assign csr_thread_masks = sched_csr_if.thread_masks;
    assign csr_mscratch     = sched_csr_if.mscratch;
    assign csr_cta_csrs     = sched_csr_if.cta_csrs;
    assign csr_cta_lane     = sched_csr_if.cta_lane;
    assign csr_mstatus      = sched_csr_if.csr_mstatus;
    assign csr_mtvec        = sched_csr_if.csr_mtvec;
    assign csr_mepc         = sched_csr_if.csr_mepc;
    assign csr_mcause       = sched_csr_if.csr_mcause;
    assign csr_mtval        = sched_csr_if.csr_mtval;
    assign sched_csr_if.csr_rd_wid        = csr_rd_wid;
    assign sched_csr_if.csr_rd_cta_id     = csr_rd_cta_id;
    assign sched_csr_if.csr_wr_valid      = csr_wr_valid;
    assign sched_csr_if.csr_wr_wid        = csr_wr_wid;
    assign sched_csr_if.csr_wr_data       = csr_wr_data;
    assign sched_csr_if.trap_csr_wr_valid = trap_csr_wr_valid;
    assign sched_csr_if.trap_csr_wr_addr  = trap_csr_wr_addr;
    assign sched_csr_if.trap_csr_wr_data  = trap_csr_wr_data;

    VX_gbar_bus_if gbar_bus_if();
    assign gbar_req_valid = gbar_bus_if.req_valid;
    assign gbar_req_data  = gbar_bus_if.req_data;
    assign gbar_bus_if.req_ready = gbar_req_ready;
    assign gbar_bus_if.rsp_valid = gbar_rsp_valid;
    assign gbar_bus_if.rsp_data  = gbar_rsp_data;
    assign gbar_rsp_ready = gbar_bus_if.rsp_ready;

    VX_scheduler #(
        .INSTANCE_ID ("scheduler"),
        .CORE_ID     (CORE_ID)
    ) scheduler (
        .clk             (clk),
        .reset           (reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN (sched_perf),
    `endif
        .warp_ctl_if     (warp_ctl_if),
        .branch_ctl_if   (branch_ctl_if),
        .decode_sched_if (decode_sched_if),
        .issue_sched_if  (issue_sched_if),
        .commit_sched_if (commit_sched_if),
        .kmu_bus_if      (kmu_bus_if),
        .schedule_if     (schedule_if),
        .sched_csr_if    (sched_csr_if),
        .gbar_bus_if     (gbar_bus_if),
        .busy            (busy)
    );

endmodule
