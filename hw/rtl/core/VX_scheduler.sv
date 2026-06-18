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

module VX_scheduler import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output sched_perf_t     sched_perf,
`endif

    // inputs
    VX_warp_ctl_if.slave    warp_ctl_if,
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_sched_unlock_if.slave sched_unlock_if,  // RTU TRACE wstall release
`endif
    VX_branch_ctl_if.slave  branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS],
    VX_decode_sched_if.slave decode_sched_if,
    VX_issue_sched_if.slave issue_sched_if [`VX_CFG_ISSUE_WIDTH],
    VX_commit_sched_if.slave commit_sched_if,

    // KMU bus
    VX_kmu_bus_if.slave     kmu_bus_if,

    // outputs
    VX_schedule_if.master   schedule_if,
    VX_sched_csr_if.master  sched_csr_if,
    VX_gbar_bus_if.master   gbar_bus_if,

    // status
    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    reg [`VX_CFG_NUM_WARPS-1:0] active_warps, active_warps_n; // updated when a warp is activated or disabled
    reg [`VX_CFG_NUM_WARPS-1:0] stalled_warps, stalled_warps_n; // set when branch/gpgpu instructions are issued

    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] thread_masks, thread_masks_n;
    reg [`VX_CFG_NUM_WARPS-1:0][PC_BITS-1:0] warp_pcs, warp_pcs_n;

    // SCS schedulable splits (mirrors the validated SimX model).
    //  - pending: the masked-off (e.g. lock-acquiring) lanes of the CURRENT loop;
    //    cancellable (merged back if the loop reconverges), else committed on
    //    vx_yield / subgroup-exit. Held in small per-warp flops.
    //  - pool: a per-warp round-robin FIFO of committed parked subgroups
    //    {tmask, pc}, stored in BRAM (VX_dp_ram) per the BRAM-first rule. Exactly
    //    one SCS event arrives per cycle, so a single 1W1R port serves all warps:
    //    a push writes the tail combinationally; a pop registers the head read and
    //    installs the {tmask,pc} one cycle later (the warp is parked meanwhile).
    //    This keeps the pool off the warp-PC critical path (was a combinational
    //    read-after-write over a large FF array → 300 MHz timing failure).
    localparam CS_W     = `VX_CFG_NUM_THREADS + PC_BITS;     // {tmask, pc}
    localparam CS_DEPTH = 2 * `VX_CFG_NUM_THREADS;           // pow2; headroom over per-lane splits
    localparam CS_SLOTW = `CLOG2(CS_DEPTH);                  // ring slot index
    localparam CS_CW    = `CLOG2(CS_DEPTH+1);                // occupancy count
    localparam CS_AW    = NW_WIDTH + CS_SLOTW;               // pool address {wid, slot}
    reg [`VX_CFG_NUM_WARPS-1:0] cs_pend, cs_pend_n;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] cs_ptmask, cs_ptmask_n;
    reg [`VX_CFG_NUM_WARPS-1:0][PC_BITS-1:0] cs_ppc, cs_ppc_n;
    reg [`VX_CFG_NUM_WARPS-1:0][CS_SLOTW-1:0] cs_head, cs_head_n;
    reg [`VX_CFG_NUM_WARPS-1:0][CS_CW-1:0] cs_cnt, cs_cnt_n;
    // lanes that have permanently exited the kernel (TMC→0); never reactivated.
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] cs_done, cs_done_n;
    // lanes currently committed to the pool (running as another split); a mask
    // restore (rs2) must not re-add them — that double-runs a work-item.
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] cs_inpool, cs_inpool_n;

    // SCS pool BRAM control + 1-cycle pop pipeline (install after registered read)
    logic                cs_we;
    logic [CS_AW-1:0]    cs_waddr, cs_raddr;
    logic [CS_W-1:0]     cs_wdata;
    wire  [CS_W-1:0]     cs_rdata;
    logic                cs_pop_set;
    logic [NW_WIDTH-1:0] cs_pop_wid;
    reg                  cs_pop_valid_r;
    reg [NW_WIDTH-1:0]   cs_pop_wid_r;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch_r;

    // Per-warp machine-mode trap CSRs. csrw writes arrive on
    // sched_csr_if.trap_csr_wr_*; ECALL/EBREAK hardware-write mepc/mcause/
    // mtval; MRET restores the warp PC from mepc.
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_XLEN-1:0] mstatus_r, mtvec_r, mepc_r, mcause_r, mtval_r;

    wire [NW_WIDTH-1:0]     schedule_wid;
    wire [`VX_CFG_NUM_THREADS-1:0] schedule_tmask;
    wire [PC_BITS-1:0]      schedule_pc;
    wire                    schedule_valid;
    wire                    schedule_ready;

    // CTA dispatcher
    wire cta_fire;
    wire [NW_WIDTH-1:0] cta_wid;
    wire [PC_BITS-1:0] cta_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] cta_tmask;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param;
    wire cta_dispatcher_busy;
    wire cta_init;

    // CTA-CSR read-back from the dispatcher (it owns the per-CTA/per-warp tables).
    cta_csrs_t                                              cta_rd_csrs;
    cta_lane_t [`VX_CFG_NUM_THREADS-1:0] cta_rd_lane;
    wire [NCTA_WIDTH-1:0]                                   schedule_cta_id;

    // SCS parked-split pool: per-warp round-robin FIFO {tmask,pc} in BRAM.
    // 1W1R, addressed by {wid, slot}; registered read gives the 1-cycle pop.
    VX_dp_ram #(
        .DATAW     (CS_W),
        .SIZE      (`VX_CFG_NUM_WARPS * CS_DEPTH),
        .RDW_MODE  ("R"),
        .OUT_REG   (1)
    ) cs_pool_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cs_we),
        .wren  (1'b1),
        .waddr (cs_waddr),
        .wdata (cs_wdata),
        .raddr (cs_raddr),
        .rdata (cs_rdata)
    );

    // Warp retirement: a TMC with tmask==0 ends the running split, but under SCS
    // the warp only truly retires (and frees its CTA slot) when it has no more
    // schedulable work — no surviving pending acquirers and an empty pool. This
    // mirrors the tmc-exit "deactivate" branch below; signalling done early would
    // let the CTA dispatcher reuse the warp mid-pool-drain and drop a parked
    // subgroup (e.g. a lock holder), hanging the kernel.
    wire [`VX_CFG_NUM_THREADS-1:0] cta_exit_mask = cs_done[warp_ctl_if.wid] | thread_masks[warp_ctl_if.wid];
    wire cta_pend_survivor = cs_pend[warp_ctl_if.wid]
                          && ((cs_ptmask[warp_ctl_if.wid] & ~cta_exit_mask) != 0);
    wire cta_warp_done = warp_ctl_if.tmc_valid && (warp_ctl_if.tmc.tmask == 0)
                      && !cta_pend_survivor && (cs_cnt[warp_ctl_if.wid] == 0);

    VX_cta_dispatch #(
        .INSTANCE_ID (`SFORMATF(("%s-cta_dispatch", INSTANCE_ID)))
    ) cta_dispatcher (
        .clk        (clk),
        .reset      (reset),
        .kmu_bus_if (kmu_bus_if),
        .active_warps(active_warps),
        .warp_done  (cta_warp_done),
        .warp_done_wid(warp_ctl_if.wid),
        .cta_fire   (cta_fire),
        .cta_wid    (cta_wid),
        .cta_PC     (cta_PC),
        .cta_tmask  (cta_tmask),
        .cta_param  (cta_param),
        .cta_init   (cta_init),
        .csr_rd_wid (sched_csr_if.csr_rd_wid),
        .csr_rd_cta_id(sched_csr_if.csr_rd_cta_id),
        .cta_rd_csrs(cta_rd_csrs),
        .cta_rd_lane(cta_rd_lane),
        .schedule_wid(schedule_wid),
        .schedule_cta_id(schedule_cta_id),
        .busy       (cta_dispatcher_busy)
    );

    assign sched_csr_if.cta_csrs = cta_rd_csrs;
    assign sched_csr_if.cta_lane = cta_rd_lane;

    assign sched_csr_if.mscratch  = mscratch_r[sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mstatus = mstatus_r[sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mtvec   = mtvec_r  [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mepc    = mepc_r   [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mcause  = mcause_r [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mtval   = mtval_r  [sched_csr_if.csr_rd_wid];

    // split/join
    wire                    join_valid;
    wire                    join_is_dvg;
    wire                    join_is_else;
    wire [NW_WIDTH-1:0]     join_wid;
    wire [`VX_CFG_NUM_THREADS-1:0] join_tmask;
    wire [PC_BITS-1:0]      join_pc;

    reg [PERF_CTR_BITS-1:0] cycles;

    wire schedule_fire = schedule_valid && schedule_ready;
    wire schedule_if_fire = schedule_if.valid && schedule_if.ready;
`ifdef VX_CFG_EXT_C_ENABLE
    // PC advance is driven by decompress_finished under EXT_C;
    `UNUSED_VAR (schedule_if_fire)
`endif

    // branch
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]               branch_valid;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][NW_WIDTH-1:0] branch_wid;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]               branch_taken;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][PC_BITS-1:0]  branch_dest;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]               branch_is_trap;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0]               branch_is_mret;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][3:0]          branch_trap_cause;
    for (genvar i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin : g_branch_init
        assign branch_valid[i]      = branch_ctl_if[i].valid;
        assign branch_wid[i]        = branch_ctl_if[i].wid;
        assign branch_taken[i]      = branch_ctl_if[i].taken;
        assign branch_dest[i]       = branch_ctl_if[i].dest;
        assign branch_is_trap[i]    = branch_ctl_if[i].is_trap;
        assign branch_is_mret[i]    = branch_ctl_if[i].is_mret;
        assign branch_trap_cause[i] = branch_ctl_if[i].trap_cause;
    end

    // barriers
    wire [`VX_CFG_NUM_WARPS-1:0] bar_unlock_mask;
    wire bar_unlock_valid;

    // wspawn
    wspawn_t wspawn;
    reg wspawn_valid;
    reg [NW_WIDTH-1:0] wspawn_wid;
    reg is_single_warp;

    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] active_warps_cnt;
    `POP_COUNT(active_warps_cnt, active_warps);

     always @(*) begin
        active_warps_n  = active_warps;
        stalled_warps_n = stalled_warps;
        thread_masks_n  = thread_masks;
        warp_pcs_n      = warp_pcs;
        cs_pend_n       = cs_pend;
        cs_ptmask_n     = cs_ptmask;
        cs_ppc_n        = cs_ppc;
        cs_head_n       = cs_head;
        cs_cnt_n        = cs_cnt;
        cs_done_n       = cs_done;
        cs_inpool_n     = cs_inpool;

        cs_we      = 1'b0;
        cs_waddr   = '0;
        cs_wdata   = '0;
        cs_raddr   = '0;
        cs_pop_set = 1'b0;
        cs_pop_wid = '0;

        // SCS: install a popped split one cycle after its BRAM read fires. The
        // warp was parked at pop-issue; reactivate it at the resumed {tmask,pc}.
        if (cs_pop_valid_r) begin
            thread_masks_n[cs_pop_wid_r] = cs_rdata[PC_BITS +: `VX_CFG_NUM_THREADS] & ~cs_done[cs_pop_wid_r];
            warp_pcs_n[cs_pop_wid_r]     = cs_rdata[PC_BITS-1:0];
            active_warps_n[cs_pop_wid_r] = 1;
            stalled_warps_n[cs_pop_wid_r] = 0; // release the park (set at pop-issue)
            cs_inpool_n[cs_pop_wid_r]    = cs_inpool[cs_pop_wid_r] & ~cs_rdata[PC_BITS +: `VX_CFG_NUM_THREADS];
        end

        // dispatch warps
        if (cta_fire) begin
            active_warps_n[cta_wid] = 1;
            // Reusing a warp for the next CTA skips the one-time prologue and rewinds to the
            // kernel's per-CTA dispatch window: a fixed 20-byte (5-instruction) sequence that
            // reloads the entry pointer and kargs before re-calling.
            warp_pcs_n[cta_wid] = cta_init ? cta_PC : (warp_pcs[cta_wid] - from_fullPC(`VX_CFG_XLEN'(20)));
            thread_masks_n[cta_wid] = cta_tmask;
            // SCS: reset per-warp split state for the (re)dispatched CTA.
            cs_pend_n[cta_wid]   = 0;
            cs_cnt_n[cta_wid]    = '0;
            cs_head_n[cta_wid]   = '0;
            cs_done_n[cta_wid]   = '0;
            cs_inpool_n[cta_wid] = '0;
        end

        // decode unlock
        if (decode_sched_if.valid && decode_sched_if.unlock) begin
            stalled_warps_n[decode_sched_if.wid] = 0;
        end

        // wspawn handling
        if (wspawn_valid && is_single_warp) begin
            active_warps_n |= wspawn.wmask;
            for (integer i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin
                if (wspawn.wmask[i] && (NW_WIDTH'(i) != wspawn_wid)) begin
                    thread_masks_n[i][0] = 1;
                    warp_pcs_n[i] = wspawn.pc;
                end
            end
            stalled_warps_n[wspawn_wid] = 0; // unlock warp
        end

        // SCS: TMC — either a normal mask set, or kernel-exit of the running
        // split. On exit, record its lanes as done, then run the next runnable
        // subgroup: pending acquirers (if any) directly, else the oldest pooled
        // split (popped via BRAM, installed next cycle), else retire the warp.
        if (warp_ctl_if.tmc_valid) begin
            if (warp_ctl_if.tmc.tmask == 0) begin
                // Record exited lanes ONLY while the warp has schedulable parked
                // work — cs_done exists solely to stop a resuming parked subgroup
                // from resurrecting a lane that already left the kernel. Marking it
                // unconditionally mis-reads a plain mask-narrowing TMC (the legacy
                // vx_spawn_threads work loop narrows then re-widens each wave) as a
                // permanent exit, filtering the re-widen to nothing. With nothing
                // parked there is nothing to protect, so TMC is honoured verbatim.
                if (cs_pend[warp_ctl_if.wid] || (cs_cnt[warp_ctl_if.wid] != 0)
                 || (cs_inpool[warp_ctl_if.wid] != 0)) begin
                    cs_done_n[warp_ctl_if.wid] = cs_done[warp_ctl_if.wid] | thread_masks[warp_ctl_if.wid];
                end
                cs_pend_n[warp_ctl_if.wid] = 0;
                if (cs_pend[warp_ctl_if.wid]
                 && (cs_ptmask[warp_ctl_if.wid] & ~cs_done_n[warp_ctl_if.wid]) != 0) begin
                    thread_masks_n[warp_ctl_if.wid] = cs_ptmask[warp_ctl_if.wid] & ~cs_done_n[warp_ctl_if.wid];
                    warp_pcs_n[warp_ctl_if.wid]     = cs_ppc[warp_ctl_if.wid];
                    active_warps_n[warp_ctl_if.wid] = 1;
                end else if (cs_cnt[warp_ctl_if.wid] != 0) begin
                    cs_pop_set = 1;
                    cs_pop_wid = warp_ctl_if.wid;
                    cs_raddr   = {warp_ctl_if.wid, cs_head[warp_ctl_if.wid]};
                    cs_head_n[warp_ctl_if.wid] = cs_head[warp_ctl_if.wid] + CS_SLOTW'(1);
                    cs_cnt_n[warp_ctl_if.wid]  = cs_cnt[warp_ctl_if.wid] - CS_CW'(1);
                    // Keep the warp ACTIVE (it still owns pooled work) but stalled
                    // until the pop installs next cycle, so the CTA dispatcher never
                    // sees a free slot mid-drain and reuses the warp.
                    active_warps_n[warp_ctl_if.wid] = 1;
                end else begin
                    active_warps_n[warp_ctl_if.wid] = 0;
                    thread_masks_n[warp_ctl_if.wid] = '0;
                end
            end else begin
                // mask set / rs2 restore: never re-add pooled or exited lanes.
                thread_masks_n[warp_ctl_if.wid] = warp_ctl_if.tmc.tmask & ~cs_inpool[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = ((warp_ctl_if.tmc.tmask & ~cs_inpool[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]) != 0);
            end
            // unlock — unless we issued a pop this cycle (warp stays parked until install)
            if (!cs_pop_set) stalled_warps_n[warp_ctl_if.wid] = 0;
        end

        // SCS: vx_pred masked off lanes — record them as the warp's (cancellable)
        // pending split, resuming at its (already +4) PC. ACCUMULATE into the
        // pending mask: a divergent loop peels lanes off across several iterations
        // (staggered exits), each a separate park at the same loop-exit PC; they
        // must all rejoin at reconvergence. Overwriting would drop the earlier
        // ones (lost work-items). This collapses the SimX multi-entry pending list
        // into one mask — valid because pred-restore ORs and clears all pending at
        // once. cs_pend is cleared between distinct loops (by restore) and in the
        // lock pattern (by vx_yield committing pending to the pool), so a fresh
        // park there correctly starts a new mask.
        if (warp_ctl_if.pred_park_valid) begin
            cs_ptmask_n[warp_ctl_if.wid] = (cs_pend[warp_ctl_if.wid] ? cs_ptmask[warp_ctl_if.wid]
                                                                     : '0)
                                         | warp_ctl_if.pred_park_tmask;
            cs_pend_n[warp_ctl_if.wid]   = 1;
            cs_ppc_n[warp_ctl_if.wid]    = warp_pcs[warp_ctl_if.wid];
        end

        // SCS: vx_pred reconverged — restore participants as the CURRENT subgroup ∪
        // the lanes THIS loop parked, never the stale rs2 (csrr-tmask) snapshot.
        // Reabsorb pending ONLY when it belongs to this very loop: a park records
        // its resume PC (cs_ppc = the pred's PC+4); the same pred's reconvergence
        // returns to that PC, so cs_ppc == warp_pcs here. A different (e.g. inner)
        // loop's reconvergence must NOT absorb lanes parked by an outer loop — they
        // would resume at the wrong PC with garbage registers (raycast misalign);
        // they stay parked until their own loop reconverges. With no matching park
        // the current subgroup alone is the correct reconvergence.
        if (warp_ctl_if.pred_restore_valid) begin
            if (cs_pend[warp_ctl_if.wid] && (cs_ppc[warp_ctl_if.wid] == warp_pcs[warp_ctl_if.wid])) begin
                thread_masks_n[warp_ctl_if.wid] = (thread_masks[warp_ctl_if.wid] | cs_ptmask[warp_ctl_if.wid]) & ~cs_done[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = ((thread_masks[warp_ctl_if.wid] | cs_ptmask[warp_ctl_if.wid]) & ~cs_done[warp_ctl_if.wid]) != 0;
                cs_pend_n[warp_ctl_if.wid]      = 0;
            end else begin
                thread_masks_n[warp_ctl_if.wid] = thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = (thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]) != 0;
            end
        end

        // split handling
        if (warp_ctl_if.split_valid) begin
            if (warp_ctl_if.split.is_dvg) begin
                thread_masks_n[warp_ctl_if.wid] = warp_ctl_if.split.then_tmask;
            end
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end

        // join handling
        if (join_valid) begin
            if (join_is_dvg) begin
                if (join_is_else) begin
                    warp_pcs_n[join_wid] = join_pc;
                end
                thread_masks_n[join_wid] = join_tmask;
            end
            stalled_warps_n[join_wid] = 0; // unlock warp
        end

        // barrier unlock handling
        if (bar_unlock_valid) begin
            stalled_warps_n &= ~bar_unlock_mask;
        end

        // wsync unlock: warp pipeline drained
        if (warp_ctl_if.wsync_valid) begin
            stalled_warps_n[warp_ctl_if.wid] = 0;
        end

        // SCS: vx_yield — defer the running (spinning) split and run the next
        // runnable one so a lock holder makes progress while spinners wait.
        // Pending acquirers run immediately; else rotate to the oldest pooled
        // split. The pool is 1W1R, so pushing the current split at the tail and
        // popping the head happen in the same cycle (distinct slots). No-op when
        // nothing else is runnable.
        if (warp_ctl_if.yield_valid) begin
            cs_pend_n[warp_ctl_if.wid] = 0;
            if (cs_pend[warp_ctl_if.wid]
             && (cs_ptmask[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]) != 0) begin
                // defer current to the tail, run pending acquirers directly
                cs_we    = 1;
                cs_waddr = {warp_ctl_if.wid, CS_SLOTW'(cs_head[warp_ctl_if.wid] + cs_cnt[warp_ctl_if.wid])};
                cs_wdata = {thread_masks[warp_ctl_if.wid], warp_pcs[warp_ctl_if.wid]};
                cs_cnt_n[warp_ctl_if.wid]    = cs_cnt[warp_ctl_if.wid] + CS_CW'(1);
                cs_inpool_n[warp_ctl_if.wid] = cs_inpool[warp_ctl_if.wid] | (thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]);
                thread_masks_n[warp_ctl_if.wid] = cs_ptmask[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid];
                warp_pcs_n[warp_ctl_if.wid]     = cs_ppc[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = 1;
            end else if (cs_cnt[warp_ctl_if.wid] != 0) begin
                // defer current to the tail, pop the oldest pooled split (head);
                // installed next cycle by the pop pipeline. Net count unchanged.
                cs_we    = 1;
                cs_waddr = {warp_ctl_if.wid, CS_SLOTW'(cs_head[warp_ctl_if.wid] + cs_cnt[warp_ctl_if.wid])};
                cs_wdata = {thread_masks[warp_ctl_if.wid], warp_pcs[warp_ctl_if.wid]};
                cs_inpool_n[warp_ctl_if.wid] = cs_inpool[warp_ctl_if.wid] | (thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]);
                cs_pop_set = 1;
                cs_pop_wid = warp_ctl_if.wid;
                cs_raddr   = {warp_ctl_if.wid, cs_head[warp_ctl_if.wid]};
                cs_head_n[warp_ctl_if.wid] = cs_head[warp_ctl_if.wid] + CS_SLOTW'(1);
                cs_cnt_n[warp_ctl_if.wid]  = cs_cnt[warp_ctl_if.wid];
                // stay ACTIVE (owns pooled work) but stalled until install, so the
                // CTA dispatcher never sees a free slot during the pop park window.
                active_warps_n[warp_ctl_if.wid] = 1;
            end
            // unlock — unless we issued a pop this cycle (parked until install)
            if (!cs_pop_set) stalled_warps_n[warp_ctl_if.wid] = 0;
        end

        // Branch handling
        for (integer i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin
            if (branch_valid[i]) begin
                if (branch_is_trap[i]) begin
                    // ECALL/EBREAK: redirect to trap vector (mtvec[1:0] = MODE field; mask off to get base address).
                    warp_pcs_n[branch_wid[i]] = from_fullPC(mtvec_r[branch_wid[i]] & ~`VX_CFG_XLEN'(3));
                end else if (branch_is_mret[i]) begin
                    // MRET/SRET/URET: restore the saved PC from mepc. ECALL/EBREAK
                    // are the only traps and they do not narrow the tmask, so there
                    // is nothing to restore beyond the PC.
                    warp_pcs_n[branch_wid[i]] = from_fullPC(mepc_r[branch_wid[i]]);
                end else if (branch_taken[i]) begin
                    warp_pcs_n[branch_wid[i]] = branch_dest[i];
                end
                stalled_warps_n[branch_wid[i]] = 0; // unlock warp
            end
        end

        // stall the warp until decode stage
        if (schedule_fire) begin
            stalled_warps_n[schedule_wid] = 1;
        end

        // advance PC.
    `ifdef VX_CFG_EXT_C_ENABLE
        // With RVC, the decompressor may emit a 2-byte instruction. Advance
        // from the committed warp PC rather than the redirect-muxed next-PC:
        // every redirect source (branch/trap/mret, split/join, wspawn) stalls
        // its warp from schedule until it resolves, so no redirect writes a
        // warp's PC on the same cycle that warp decode-advances. Reading the
        // registered PC keeps the +2/+4 adder off the branch/trap redirect
        // cone, matching the non-RVC path which advances the pipeline-carried
        // PC instead of the combinational next-PC.
        if (decode_sched_if.valid) begin
            warp_pcs_n[decode_sched_if.wid] =
                warp_pcs[decode_sched_if.wid]
                + from_fullPC(decode_sched_if.is_rvc ? `VX_CFG_XLEN'(2) : `VX_CFG_XLEN'(4));
        end
    `else
        if (schedule_if_fire) begin
            warp_pcs_n[schedule_if.data.wid] = schedule_if.data.PC + from_fullPC(`VX_CFG_XLEN'(4));
        end
    `endif

    `ifdef VX_CFG_EXT_RTU_ENABLE
        // A wstall'd TRACE retires (its traversal's first response landed and the
        // arm op wrote back the handle): resume the warp so it proceeds to WAIT,
        // which returns the response status (terminal or candidate). No trap, no
        // redirect — the candidate is serviced inline by the warp's loop.
        if (sched_unlock_if.valid) begin
            stalled_warps_n[sched_unlock_if.wid] = 1'b0;
        end
    `endif
    end

    always @(posedge clk) begin
        if (reset) begin
            stalled_warps   <= '0;
            warp_pcs        <= '0;
            active_warps    <= '0;
            thread_masks    <= '0;
            cycles          <= '0;
            wspawn_valid    <=  0;
            warp_pcs        <= '0;
            active_warps    <= '0;
            thread_masks    <= '0;
            is_single_warp  <= 0;
            mscratch_r      <= '0;
            mstatus_r       <= '0;
            mtvec_r         <= '0;
            mepc_r          <= '0;
            mcause_r        <= '0;
            mtval_r         <= '0;
            cs_pend         <= '0;
            cs_cnt          <= '0;
            cs_head         <= '0;
            cs_done         <= '0;
            cs_inpool       <= '0;
            cs_pop_valid_r  <= '0;
        end else begin
            active_warps   <= active_warps_n;
            stalled_warps  <= stalled_warps_n;
            thread_masks   <= thread_masks_n;
            warp_pcs       <= warp_pcs_n;
            cs_pend        <= cs_pend_n;
            cs_ptmask      <= cs_ptmask_n;
            cs_ppc         <= cs_ppc_n;
            cs_head        <= cs_head_n;
            cs_cnt         <= cs_cnt_n;
            cs_done        <= cs_done_n;
            cs_inpool      <= cs_inpool_n;
            cs_pop_valid_r <= cs_pop_set;
            cs_pop_wid_r   <= cs_pop_wid;
            is_single_warp <= (active_warps_cnt == $bits(active_warps_cnt)'(1));

            // wspawn handling
            if (warp_ctl_if.wspawn_valid) begin
                wspawn_valid <= 1;
                wspawn.wmask <= warp_ctl_if.wspawn.wmask;
                wspawn.pc    <= warp_ctl_if.wspawn.pc;
                wspawn_wid   <= warp_ctl_if.wid;
            end
            if (wspawn_valid && is_single_warp) begin
                wspawn_valid <= 0;
                // copy mscratch from spawning warp to all newly spawned warps
                for (integer i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin
                    if (wspawn.wmask[i] && (NW_WIDTH'(i) != wspawn_wid)) begin
                        mscratch_r[i] <= mscratch_r[wspawn_wid];
                    end
                end
            end

            // CTA dispatch: latch this warp's mscratch (param). The per-CTA /
            // per-warp tables and the wid->cta_id map live in VX_cta_dispatch.
            if (cta_fire) begin
                mscratch_r[cta_wid] <= cta_param;
            end

            // MSCRATCH write-back from CSR unit (CSR instruction)
            if (sched_csr_if.csr_wr_valid) begin
                mscratch_r[sched_csr_if.csr_wr_wid] <= sched_csr_if.csr_wr_data;
            end

            // Trap CSR write-back from CSR unit (csrw mstatus/mtvec/mepc/...)
            if (sched_csr_if.trap_csr_wr_valid) begin
                case (sched_csr_if.trap_csr_wr_addr)
                    `VX_CSR_MSTATUS: mstatus_r[sched_csr_if.csr_wr_wid] <= sched_csr_if.trap_csr_wr_data;
                    `VX_CSR_MTVEC:   mtvec_r  [sched_csr_if.csr_wr_wid] <= sched_csr_if.trap_csr_wr_data;
                    `VX_CSR_MEPC:    mepc_r   [sched_csr_if.csr_wr_wid] <= sched_csr_if.trap_csr_wr_data;
                    `VX_CSR_MCAUSE:  mcause_r [sched_csr_if.csr_wr_wid] <= sched_csr_if.trap_csr_wr_data;
                    `VX_CSR_MTVAL:   mtval_r  [sched_csr_if.csr_wr_wid] <= sched_csr_if.trap_csr_wr_data;
                    default:;
                endcase
            end

            // Hardware trap entry (ECALL/EBREAK): snapshot the faulting PC
            // into mepc and the cause into mcause. Ordered after the
            // software write so a hardware trap wins a same-cycle conflict.
            for (integer i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin
                if (branch_valid[i] && branch_is_trap[i]) begin
                    mepc_r  [branch_wid[i]] <= to_fullPC(branch_dest[i]);
                    mcause_r[branch_wid[i]] <= `VX_CFG_XLEN'(branch_trap_cause[i]);
                    mtval_r [branch_wid[i]] <= '0;
                end
            end

            if (busy) begin
                cycles <= cycles + 1;
            end
        end
    end

    // Barrier unit

    VX_bar_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-barrier", INSTANCE_ID))),
        .CORE_ID     (CORE_ID)
    ) bar_unit (
        .clk        (clk),
        .reset      (reset),
        .req_valid  (warp_ctl_if.bar_valid),
        .req_wid    (warp_ctl_if.wid),
        .req_data   (warp_ctl_if.bar),
        .read_addr  (warp_ctl_if.bar_addr),
        .read_phase (warp_ctl_if.bar_phase),
        .active_warps(active_warps),
        .gbar_bus_if(gbar_bus_if),
        .unlock_valid(bar_unlock_valid),
        .unlock_mask(bar_unlock_mask)
    );

    // split/join handling

    VX_split_join #(
        .INSTANCE_ID (`SFORMATF(("%s-splitjoin", INSTANCE_ID))),
        .OUT_REG     (1)
    ) split_join (
        .clk        (clk),
        .reset      (reset),
        .split_valid(warp_ctl_if.split_valid),
        .sjoin_valid(warp_ctl_if.sjoin_valid),
        .wid        (warp_ctl_if.wid),
        .split      (warp_ctl_if.split),
        .sjoin      (warp_ctl_if.sjoin),
        .join_valid (join_valid),
        .join_is_dvg(join_is_dvg),
        .join_is_else(join_is_else),
        .join_wid   (join_wid),
        .join_tmask (join_tmask),
        .join_pc    (join_pc),
        .stack_wid  (warp_ctl_if.dvstack_wid),
        .stack_ptr  (warp_ctl_if.dvstack_ptr)
    );

    // schedule the next ready warp

    wire [`VX_CFG_NUM_WARPS-1:0] ready_warps = active_warps & ~stalled_warps;

    // Per-warp ibuffer occupancy counter (registered full[i] keeps arbitration
    // off the critical path; full_n feeds an externally registered aggregate
    // so all_full is valid the same cycle as full[i]).
    localparam IBUF_CW = $clog2(`VX_CFG_IBUF_SIZE + 1);

    wire [`VX_CFG_NUM_WARPS-1:0] schedule_onehot;
    logic [`VX_CFG_NUM_WARPS-1:0] ibuf_full, ibuf_full_n;

    for (genvar i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin : g_ibuf_cnt
        logic [IBUF_CW-1:0] size_r, size_n;
        wire incr = schedule_fire && schedule_onehot[i];
        wire decr = schedule_if.ibuf_pop[i];
        assign size_n = size_r + IBUF_CW'(incr) - IBUF_CW'(decr);
        assign ibuf_full_n[i] = (size_n == IBUF_CW'(`VX_CFG_IBUF_SIZE));
        always @(posedge clk) begin
            if (reset) begin
                size_r       <= '0;
                ibuf_full[i] <= 1'b0;
            end else begin
                size_r       <= size_n;
                ibuf_full[i] <= ibuf_full_n[i];
            end
        end
    end

    wire [`VX_CFG_NUM_WARPS-1:0] preferred_warps = ready_warps & ~ibuf_full;
`ifndef L1_ENABLE
    // without L1, we should ensure the icache never stalls,
    // because it could deadlock dcache response since they share the same bus.
    wire [`VX_CFG_NUM_WARPS-1:0] schedule_warps = preferred_warps;
`else
    reg all_ibuf_full;
    always @(posedge clk) begin
        if (reset) all_ibuf_full <= 1'b0;
        else all_ibuf_full <= (& ibuf_full_n);
    end
    wire [`VX_CFG_NUM_WARPS-1:0] schedule_warps = all_ibuf_full ? ready_warps : preferred_warps;
`endif

    VX_priority_encoder #(
        .N (`VX_CFG_NUM_WARPS)
    ) wid_select (
        .data_in   (schedule_warps),
        .index_out (schedule_wid),
        .valid_out (schedule_valid),
        .onehot_out(schedule_onehot)
    );

    wire [`VX_CFG_NUM_WARPS-1:0][(`VX_CFG_NUM_THREADS + PC_BITS)-1:0] schedule_data;
    for (genvar i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin : g_schedule_data
        assign schedule_data[i] = {thread_masks[i], warp_pcs[i]};
    end

    assign {schedule_tmask, schedule_pc} = {
        schedule_data[schedule_wid][(`VX_CFG_NUM_THREADS + PC_BITS)-1:(`VX_CFG_NUM_THREADS + PC_BITS)-4],
        schedule_data[schedule_wid][(`VX_CFG_NUM_THREADS + PC_BITS)-5:0]
    };

    wire [UUID_WIDTH-1:0] instr_uuid;
`ifdef UUID_ENABLE
    VX_uuid_gen #(
        .CORE_ID (CORE_ID)
    ) uuid_gen (
        .clk   (clk),
        .reset (reset),
        .incr  (schedule_fire),
        .wid   (schedule_wid),
        .uuid  (instr_uuid)
    );
`else
    assign instr_uuid = '0;
`endif

    // schedule_cta_id is produced by VX_cta_dispatch from its wid->cta_id map.

    VX_elastic_buffer #(
        .DATAW (`VX_CFG_NUM_THREADS + PC_BITS + NW_WIDTH + NCTA_WIDTH + UUID_WIDTH),
        .SIZE  (2),  // need to buffer out ready_in
        .OUT_REG (1) // should be registered for BRAM acces in fetch unit
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (schedule_valid),
        .ready_in  (schedule_ready),
        .data_in   ({schedule_tmask, schedule_pc, schedule_wid, schedule_cta_id, instr_uuid}),
        .data_out  ({schedule_if.data.tmask, schedule_if.data.PC, schedule_if.data.wid, schedule_if.data.cta_id, schedule_if.data.uuid}),
        .valid_out (schedule_if.valid),
        .ready_out (schedule_if.ready)
    );

    // Track committed instructions

    reg [PERF_CTR_BITS-1:0] instret;

    wire [`VX_CFG_NUM_WARPS-1:0] committed_warps_v = commit_sched_if.committed_warps;
    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] committed_warps_cnt_v;
    `POP_COUNT(committed_warps_cnt_v, committed_warps_v);

    always @(posedge clk) begin
        if (reset) begin
            instret <= '0;
        end else begin
            instret <= instret + PERF_CTR_BITS'(committed_warps_cnt_v);
        end
    end

    // Track pending instructions per warp

    wire [`VX_CFG_NUM_WARPS-1:0] pending_warp_empty;
    wire [`VX_CFG_NUM_WARPS-1:0] pending_warp_alm_empty;

    for (genvar i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin : g_pending_warps
        localparam logic [ISSUE_ISW_W-1:0] isw = wid_to_isw(i);
        localparam logic [ISSUE_WIS_W-1:0] wis = wid_to_wis(i);

        VX_pending_size #(
            .SIZE      (256),
            .ALM_EMPTY (1)
        ) per_warp_ctr (
            .clk       (clk),
            .reset     (reset),
            .incr      (issue_sched_if[isw].valid && (issue_sched_if[isw].wis == ISSUE_WIS_W'(wis))),
            .decr      (commit_sched_if.committed_warps[i]),
            .empty     (pending_warp_empty[i]),
            .alm_empty (pending_warp_alm_empty[i]),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end

    wire busy_buf;
    `BUFFER_EX(busy_buf, (active_warps_n != 0 || ~(&pending_warp_empty)), 1'b1, 1, 1);
    assign busy = busy_buf || cta_dispatcher_busy;

    assign warp_ctl_if.warp_pending_alm_empty = pending_warp_alm_empty;

    // export CSRs
    assign sched_csr_if.cycles = cycles;
    assign sched_csr_if.instret = instret;
    assign sched_csr_if.active_warps = active_warps;
    assign sched_csr_if.thread_masks = thread_masks;

   // timeout handling
    reg [31:0] timeout_ctr;
    reg timeout_enable;
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr    <= '0;
            timeout_enable <= 0;
        end else begin
            if (decode_sched_if.valid && decode_sched_if.unlock) begin
                timeout_enable <= 1;
            end
            if (timeout_enable && active_warps !=0 && active_warps == stalled_warps) begin
                timeout_ctr <= timeout_ctr + 1;
            end else if (active_warps == 0 || active_warps != stalled_warps) begin
                timeout_ctr <= '0;
            end
        end
    end

`ifdef EXT_SCHED_STALL_TIMEOUT
    localparam SCHED_STALL_TIMEOUT = `EXT_SCHED_STALL_TIMEOUT;
`else
    localparam SCHED_STALL_TIMEOUT = STALL_TIMEOUT;
`endif
`ifdef EXT_SCHED_TIMEOUT_DUMP
    always @(posedge clk) begin
        if (!reset && (timeout_ctr == (SCHED_STALL_TIMEOUT - 1))) begin
            $display("*** %s scheduler-timeout dump: active=%b stalled=%b", INSTANCE_ID, active_warps, stalled_warps);
            for (integer wi = 0; wi < `VX_CFG_NUM_WARPS; ++wi) begin
                $display("    wid=%0d stalled=%0d pc=0x%0h tmask=%b",
                         wi, stalled_warps[wi], to_fullPC(warp_pcs[wi]), thread_masks[wi]);
            end
        end
    end
`endif
    `RUNTIME_ASSERT(timeout_ctr < SCHED_STALL_TIMEOUT, ("*** %s timeout: active_warps=%b, stalled_warps=%b", INSTANCE_ID, active_warps, stalled_warps))

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_sched_idles;
    reg [PERF_CTR_BITS-1:0] perf_active_warps;
    reg [PERF_CTR_BITS-1:0] perf_stalled_warps;
    reg [PERF_CTR_BITS-1:0] perf_issued_warps;
    reg [PERF_CTR_BITS-1:0] perf_issued_threads;
    reg [PERF_CTR_BITS-1:0] perf_branches;
    reg [PERF_CTR_BITS-1:0] perf_divergence;

    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] stalled_warps_cnt;
    wire [`CLOG2(`VX_CFG_NUM_ALU_BLOCKS+1)-1:0] branches_cnt;
    wire [`CLOG2(`VX_CFG_NUM_THREADS+1)-1:0] issued_threads_cnt;

    wire schedule_idle = ~schedule_valid;
    wire has_divergence = warp_ctl_if.split_valid && warp_ctl_if.split.is_dvg;
    wire [`VX_CFG_NUM_THREADS-1:0] issued_threads = {`VX_CFG_NUM_THREADS{schedule_if_fire}} & schedule_if.data.tmask;

    `POP_COUNT(stalled_warps_cnt, stalled_warps);
    `POP_COUNT(issued_threads_cnt, issued_threads);
    `POP_COUNT(branches_cnt, branch_valid);

    always @(posedge clk) begin
        if (reset) begin
            perf_sched_idles   <= '0;
            perf_active_warps  <= '0;
            perf_stalled_warps <= '0;
            perf_issued_warps  <= '0;
            perf_issued_threads<= '0;
            perf_branches      <= '0;
            perf_divergence    <= '0;
        end else begin
            perf_sched_idles   <= perf_sched_idles + PERF_CTR_BITS'(schedule_idle);
            perf_active_warps  <= perf_active_warps + PERF_CTR_BITS'(active_warps_cnt);
            perf_stalled_warps <= perf_stalled_warps + PERF_CTR_BITS'(stalled_warps_cnt);
            perf_issued_warps  <= perf_issued_warps + PERF_CTR_BITS'(schedule_if_fire);
            perf_issued_threads<= perf_issued_threads + PERF_CTR_BITS'(issued_threads_cnt);
            perf_branches      <= perf_branches + PERF_CTR_BITS'(branches_cnt);
            perf_divergence    <= perf_divergence + PERF_CTR_BITS'(has_divergence);
        end
    end

    assign sched_perf.idles         = perf_sched_idles;
    assign sched_perf.active_warps  = perf_active_warps;
    assign sched_perf.stalled_warps = perf_stalled_warps;
    assign sched_perf.issued_warps  = perf_issued_warps;
    assign sched_perf.issued_threads= perf_issued_threads;
    assign sched_perf.branches      = perf_branches;
    assign sched_perf.divergence    = perf_divergence;
`endif

`ifdef DBG_TRACE_PIPELINE
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_trace_warp_status
        always @(posedge clk) begin
            if (active_warps_n[w] != active_warps[w]
             || (active_warps[w] && (stalled_warps_n[w] != stalled_warps[w]
                                  || thread_masks_n[w] != thread_masks[w]))) begin
                `TRACE(1, ("%t: %s warp-state: wid=%0d, active=%b, stalled=%b, tmask=%b\n",
                    $time, INSTANCE_ID, w, active_warps_n[w], stalled_warps_n[w], thread_masks_n[w]
                ))
            end
        end
    end

    always @(posedge clk) begin
        if (schedule_fire) begin
            `TRACE(1, ("%t: %s dispatch: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_wid, schedule_cta_id, to_fullPC(schedule_pc), schedule_tmask, instr_uuid))
        end
    end
`endif

endmodule
