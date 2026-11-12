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

// Standalone synthesis wrapper for VX_scheduler in isolation, so its area,
// Fmax and BRAM (SCS pool) cost can be measured without pulling in the rest
// of VX_core_top (fetch/decode/issue/execute/lsu/fpu/caches/mmu).
//
// Every real input of VX_scheduler is promoted to a top-level input port
// (same idiom as hw/unittest/issue/VX_issue_top.sv) rather than left
// dangling: an undriven internal wire synthesizes as a compile-time
// constant, which lets Yosys constant-propagate the entire warp-scheduling
// FSM away (its whole purpose is to react to these inputs). Likewise every
// output VX_scheduler drives is XOR-reduced into `sink` so it has fanout
// past the module boundary and nothing gets removed as dead code.
module VX_scheduler_top import VX_gpu_pkg::*; (
    input wire clk,
    input wire reset,

    // warp_ctl_if (slave: scheduler receives these)
    input wire                          wc_wspawn_valid,
    input wire                          wc_tmc_valid,
    input wire                          wc_split_valid,
    input wire                          wc_sjoin_valid,
    input wire                          wc_bar_valid,
    input wire                          wc_wsync_valid,
    input wire                          wc_yield_valid,
    input wire                          wc_pred_park_valid,
    input wire [`VX_CFG_NUM_THREADS-1:0] wc_pred_park_tmask,
    input wire                          wc_pred_restore_valid,
    input wire [NW_WIDTH-1:0]           wc_wid,
    input tmc_t                         wc_tmc,
    input wspawn_t                      wc_wspawn,
    input split_t                       wc_split,
    input join_t                        wc_sjoin,
    input barrier_t                     wc_bar,
    input wire [BAR_ADDR_W-1:0]         wc_bar_addr,
    input wire [NW_WIDTH-1:0]           wc_dvstack_wid,

    // branch_ctl_if[NUM_ALU_BLOCKS] (slave: all inputs)
    input wire                          bc_valid      [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [NW_WIDTH-1:0]           bc_wid        [`VX_CFG_NUM_ALU_BLOCKS],
    input wire                          bc_taken      [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [PC_BITS-1:0]            bc_dest       [`VX_CFG_NUM_ALU_BLOCKS],
    input wire                          bc_is_trap    [`VX_CFG_NUM_ALU_BLOCKS],
    input wire                          bc_is_mret    [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [3:0]                    bc_trap_cause [`VX_CFG_NUM_ALU_BLOCKS],
`ifdef VX_CFG_DIVERGE_TYPE_SCS
    input wire                          bc_is_pbr        [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [`VX_CFG_NUM_THREADS-1:0] bc_keep_mask    [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [`VX_CFG_NUM_THREADS-1:0] bc_pbr_tmask    [`VX_CFG_NUM_ALU_BLOCKS],
    input wire [PC_BITS-1:0]            bc_pbr_ntaken_pc [`VX_CFG_NUM_ALU_BLOCKS],
`endif

    // decode_sched_if (slave: all inputs)
    input wire                          ds_valid,
    input wire                          ds_unlock,
    input wire [NW_WIDTH-1:0]           ds_wid,

    // issue_sched_if[ISSUE_WIDTH] (slave: all inputs)
    input wire                          is_valid [`VX_CFG_ISSUE_WIDTH],
    input wire [ISSUE_WIS_W-1:0]        is_wis   [`VX_CFG_ISSUE_WIDTH],

    // commit_sched_if (slave: all inputs)
    input wire [`VX_CFG_NUM_WARPS-1:0]  cs_committed_warps,

    // kmu_bus_if (slave)
    input wire                          kmu_valid,
    input wire                          kmu_kind,
    input wire                          kmu_eop,
    input wire [KMU_DEST_W-1:0]         kmu_dest,
    input wire [KMU_DATAW-1:0]          kmu_data,

    output wire sink
);

    VX_warp_ctl_if warp_ctl_if();
    VX_branch_ctl_if branch_ctl_if[`VX_CFG_NUM_ALU_BLOCKS]();
    VX_decode_sched_if decode_sched_if();
    VX_issue_sched_if issue_sched_if[`VX_CFG_ISSUE_WIDTH]();
    VX_commit_sched_if commit_sched_if();
    VX_kmu_bus_if kmu_bus_if();
    VX_schedule_if schedule_if();
    VX_sched_csr_if sched_csr_if();
    VX_gbar_bus_if gbar_bus_if();
    wire busy;

    assign warp_ctl_if.wspawn_valid       = wc_wspawn_valid;
    assign warp_ctl_if.tmc_valid          = wc_tmc_valid;
    assign warp_ctl_if.split_valid        = wc_split_valid;
    assign warp_ctl_if.sjoin_valid        = wc_sjoin_valid;
    assign warp_ctl_if.bar_valid          = wc_bar_valid;
    assign warp_ctl_if.wsync_valid        = wc_wsync_valid;
    assign warp_ctl_if.yield_valid        = wc_yield_valid;
    assign warp_ctl_if.pred_park_valid    = wc_pred_park_valid;
    assign warp_ctl_if.pred_park_tmask    = wc_pred_park_tmask;
    assign warp_ctl_if.pred_restore_valid = wc_pred_restore_valid;
    assign warp_ctl_if.wid                = wc_wid;
    assign warp_ctl_if.tmc                = wc_tmc;
    assign warp_ctl_if.wspawn             = wc_wspawn;
    assign warp_ctl_if.split              = wc_split;
    assign warp_ctl_if.sjoin              = wc_sjoin;
    assign warp_ctl_if.bar                = wc_bar;
    assign warp_ctl_if.bar_addr           = wc_bar_addr;
    assign warp_ctl_if.dvstack_wid        = wc_dvstack_wid;

    for (genvar i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin : g_branch_ctl_if
        assign branch_ctl_if[i].valid      = bc_valid[i];
        assign branch_ctl_if[i].wid        = bc_wid[i];
        assign branch_ctl_if[i].taken      = bc_taken[i];
        assign branch_ctl_if[i].dest       = bc_dest[i];
        assign branch_ctl_if[i].is_trap    = bc_is_trap[i];
        assign branch_ctl_if[i].is_mret    = bc_is_mret[i];
        assign branch_ctl_if[i].trap_cause = bc_trap_cause[i];
    `ifdef VX_CFG_DIVERGE_TYPE_SCS
        assign branch_ctl_if[i].is_pbr        = bc_is_pbr[i];
        assign branch_ctl_if[i].keep_mask     = bc_keep_mask[i];
        assign branch_ctl_if[i].pbr_tmask     = bc_pbr_tmask[i];
        assign branch_ctl_if[i].pbr_ntaken_pc = bc_pbr_ntaken_pc[i];
    `endif
    end

    assign decode_sched_if.valid  = ds_valid;
    assign decode_sched_if.unlock = ds_unlock;
    assign decode_sched_if.wid    = ds_wid;

    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_issue_sched_if
        assign issue_sched_if[i].valid = is_valid[i];
        assign issue_sched_if[i].wis   = is_wis[i];
    end

    assign commit_sched_if.committed_warps = cs_committed_warps;

    assign kmu_bus_if.valid = kmu_valid;
    assign kmu_bus_if.kind  = kmu_kind;
    assign kmu_bus_if.eop   = kmu_eop;
    assign kmu_bus_if.dest  = kmu_dest;
    assign kmu_bus_if.data  = kmu_data;

    VX_scheduler #(
        .INSTANCE_ID ("scheduler0"),
        .CORE_ID     (0)
    ) scheduler (
        .clk             (clk),
        .reset           (reset),
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

    // Every signal VX_scheduler actually drives, XOR-reduced into one bit so
    // none of it is dead code from Yosys's perspective. Purely a synthesis
    // anchor -- functional correctness of `sink` itself is irrelevant.
    assign sink =
        busy
        ^ warp_ctl_if.bar_phase
        ^ (^warp_ctl_if.warp_pending_alm_empty)
        ^ warp_ctl_if.lsu_sched_drained
        ^ (^warp_ctl_if.dvstack_ptr)
        ^ kmu_bus_if.ready
        ^ schedule_if.valid
        ^ (^schedule_if.data)
        ^ (^sched_csr_if.cycles)
        ^ (^sched_csr_if.instret)
        ^ (^sched_csr_if.active_warps)
        ^ (^sched_csr_if.thread_masks)
        ^ (^sched_csr_if.mscratch)
        ^ (^sched_csr_if.cta_csrs)
        ^ (^sched_csr_if.cta_lane)
        ^ (^sched_csr_if.csr_mstatus)
        ^ (^sched_csr_if.csr_mtvec)
        ^ (^sched_csr_if.csr_mepc)
        ^ (^sched_csr_if.csr_mcause)
        ^ (^sched_csr_if.csr_mtval)
        ^ gbar_bus_if.req_valid
        ^ (^gbar_bus_if.req_data)
        ^ gbar_bus_if.rsp_ready;

endmodule
