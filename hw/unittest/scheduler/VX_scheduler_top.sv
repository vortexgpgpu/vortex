// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

`include "VX_define.vh"

module VX_scheduler_top import VX_gpu_pkg::*; (
    input  wire clk,
    input  wire reset,

    input  wire task_valid,
    output wire task_ready,
    input  wire [CTA_TID_WIDTH:0] task_block_size,
    input  wire [7:0] task_ctx_id,

    input  wire ctl_tmc_valid,
    input  wire [NW_WIDTH-1:0] ctl_wid,
    input  wire [`VX_CFG_NUM_THREADS-1:0] ctl_tmask,
    input  wire [`VX_CFG_NUM_WARPS-1:0] group_drained_mask,

    output wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    output wire scheduler_busy,
    output wire epoch_adv_valid,
    output wire [NW_WIDTH-1:0] epoch_adv_wid,
    output wire poison_valid,
    output wire [NW_WIDTH-1:0] poison_wid
);
    VX_warp_ctl_if warp_ctl_if();
    assign warp_ctl_if.wspawn_valid = 1'b0;
    assign warp_ctl_if.tmc_valid = ctl_tmc_valid;
    assign warp_ctl_if.split_valid = 1'b0;
    assign warp_ctl_if.sjoin_valid = 1'b0;
    assign warp_ctl_if.bar_valid = 1'b0;
    assign warp_ctl_if.wsync_valid = 1'b0;
    assign warp_ctl_if.wid = ctl_wid;
    assign warp_ctl_if.wspawn = '0;
    assign warp_ctl_if.tmc.tmask = ctl_tmask;
    assign warp_ctl_if.split = '0;
    assign warp_ctl_if.sjoin = '0;
    assign warp_ctl_if.bar = '0;
    assign warp_ctl_if.bar_addr = '0;
    assign warp_ctl_if.lsu_sched_drained = 1'b1;
    assign warp_ctl_if.dvstack_wid = '0;

    VX_branch_ctl_if branch_ctl_if[`VX_CFG_NUM_ALU_BLOCKS]();
    for (genvar i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin : g_branch
        assign branch_ctl_if[i].valid = 1'b0;
        assign branch_ctl_if[i].wid = '0;
        assign branch_ctl_if[i].taken = 1'b0;
        assign branch_ctl_if[i].dest = '0;
        assign branch_ctl_if[i].is_trap = 1'b0;
        assign branch_ctl_if[i].is_mret = 1'b0;
        assign branch_ctl_if[i].trap_cause = '0;
    end

    VX_decode_sched_if decode_sched_if();
    assign decode_sched_if.valid = 1'b0;
    assign decode_sched_if.unlock = 1'b0;
    assign decode_sched_if.wid = '0;
`ifdef VX_CFG_EXT_C_ENABLE
    assign decode_sched_if.is_rvc = 1'b0;
`endif

    VX_issue_sched_if issue_sched_if[`VX_CFG_ISSUE_WIDTH]();
    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_issue
        assign issue_sched_if[i].valid = 1'b0;
        assign issue_sched_if[i].wis = '0;
    end

    VX_commit_sched_if commit_sched_if();
    assign commit_sched_if.committed_warps = '0;

    VX_kmu_bus_if kmu_bus_if();
    assign kmu_bus_if.valid = task_valid;
    always_comb begin
        kmu_bus_if.data = '0;
        kmu_bus_if.data.PC = PC_BITS'(32'h100);
        kmu_bus_if.data.entry = PC_BITS'(32'h100);
        kmu_bus_if.data.ctx_id = task_ctx_id;
        kmu_bus_if.data.block_dim[0] = task_block_size;
        kmu_bus_if.data.block_dim[1] = 1;
        kmu_bus_if.data.block_dim[2] = 1;
        kmu_bus_if.data.grid_dim[0] = 1;
        kmu_bus_if.data.grid_dim[1] = 1;
        kmu_bus_if.data.grid_dim[2] = 1;
        kmu_bus_if.data.block_size = task_block_size;
        kmu_bus_if.data.warp_step[0] = CTA_TID_WIDTH'(`VX_CFG_NUM_THREADS);
        kmu_bus_if.data.cluster_size = (NW_WIDTH+1)'(1);
        kmu_bus_if.data.is_first_of_cluster = 1'b1;
    end
    assign task_ready = kmu_bus_if.ready;

    VX_schedule_if schedule_if();
    assign schedule_if.ready = 1'b0;
    assign schedule_if.ibuf_pop = '0;

    VX_sched_csr_if sched_csr_if();
    assign sched_csr_if.csr_rd_wid = '0;
    assign sched_csr_if.csr_rd_cta_id = '0;
    assign sched_csr_if.csr_wr_valid = 1'b0;
    assign sched_csr_if.csr_wr_wid = '0;
    assign sched_csr_if.csr_wr_data = '0;
    assign sched_csr_if.trap_csr_wr_valid = 1'b0;
    assign sched_csr_if.trap_csr_wr_addr = '0;
    assign sched_csr_if.trap_csr_wr_data = '0;
`ifdef VX_CFG_VM_ENABLE
    assign sched_csr_if.csr_satp = '0;
`endif
    assign active_warps = sched_csr_if.active_warps;

    VX_gbar_bus_if gbar_bus_if();
    assign gbar_bus_if.req_ready = 1'b1;
    assign gbar_bus_if.rsp_valid = 1'b0;
    assign gbar_bus_if.rsp_data = '0;

    VX_scheduler #(
        .INSTANCE_ID ("scheduler-test"),
        .CORE_ID     (0)
    ) scheduler (
        .clk                       (clk),
        .reset                     (reset),
        .warp_ctl_if               (warp_ctl_if),
        .branch_ctl_if             (branch_ctl_if),
        .decode_sched_if           (decode_sched_if),
        .issue_sched_if            (issue_sched_if),
        .commit_sched_if           (commit_sched_if),
        .dxa_group_unlock_mask     ('0),
        .dxa_group_drained_mask    (group_drained_mask),
        .dxa_group_epoch_adv_valid (epoch_adv_valid),
        .dxa_group_epoch_adv_wid   (epoch_adv_wid),
        .dxa_group_poison_valid    (poison_valid),
        .dxa_group_poison_wid      (poison_wid),
        .kmu_bus_if                (kmu_bus_if),
        .schedule_if               (schedule_if),
        .sched_csr_if              (sched_csr_if),
        .gbar_bus_if               (gbar_bus_if),
        .busy                      (scheduler_busy)
    );

    wire unused_outputs = warp_ctl_if.bar_phase
                        || (|warp_ctl_if.warp_pending_alm_empty)
                        || warp_ctl_if.lsu_sched_drained
                        || (|warp_ctl_if.dvstack_ptr)
                        || schedule_if.valid
                        || (|schedule_if.data)
                        || gbar_bus_if.req_valid
                        || (|gbar_bus_if.req_data)
                        || gbar_bus_if.rsp_ready
                        || (|sched_csr_if.cycles)
                        || (|sched_csr_if.instret)
                        || (|sched_csr_if.thread_masks)
                        || (|sched_csr_if.mscratch)
                        || (|sched_csr_if.cta_csrs)
                        || (|sched_csr_if.cta_tid)
                        || (|sched_csr_if.csr_mstatus)
                        || (|sched_csr_if.csr_mtvec)
                        || (|sched_csr_if.csr_mepc)
                        || (|sched_csr_if.csr_mcause)
                        || (|sched_csr_if.csr_mtval);
    `UNUSED_VAR (unused_outputs)

endmodule
