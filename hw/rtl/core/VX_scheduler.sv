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

`ifdef VX_CFG_DIVERGE_TYPE_SCS
`ifdef THREADSPLIT_EVAL
    // ThreadSplit A/B: +threadsplit_ipdom disables the SCS machinery at runtime.
    reg scs_enabled;
    initial scs_enabled = !$test$plusargs("threadsplit_ipdom");
`else
    wire scs_enabled = 1'b1;
`endif

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
`ifdef VX_CFG_SCS_POOL_DEPTH
    // Case-study capacity sweep: override the resident pool depth K independently
    // of warp width, to measure how few schedulable contexts SCS actually needs.
    localparam CS_DEPTH = `VX_CFG_SCS_POOL_DEPTH;
`else
    localparam CS_DEPTH = `VX_CFG_NUM_THREADS;               // one resident pool slot per lane
`endif
    // CLOG2(1) == 0, which would declare zero-width head/address vectors; a
    // 1-entry pool still needs a 1-bit ring index (it just alternates 0/1 while
    // holding at most one entry — see the BRAM sizing note below).
    localparam CS_SLOTW = (CS_DEPTH > 1) ? `CLOG2(CS_DEPTH) : 1;
    localparam CS_CW    = `CLOG2(CS_DEPTH+1);                // occupancy count (0..CS_DEPTH)
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
`endif // VX_CFG_DIVERGE_TYPE_SCS
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    // ITS (mirror of the SimX NV_ITS model): threads diverge on per-thread
    // PCs, regrouped through convergence barriers and the Yielded state.
    // Storage is memory-first: parked per-thread PCs and the barrier masks
    // live in per-warp LUTRAM rows owned by a single regroup engine, and the
    // schedule path reads only the registered per-warp group cache
    // {grp_pc, grp_mask} - the same shape as the baseline warp_pcs read.
    //  - a thread outside its warp's current group has its true PC in the
    //    row store; group members' truth is grp_pc (rows go stale until the
    //    next park write).
    //  - group-changing events (branch resolution; bar_add/bar_wait/yield/
    //    tmc; CTA/wspawn init; yield wake) are serialized one per cycle into
    //    the engine, which applies the row writes, re-evaluates barrier
    //    release, and rebuilds the group through one min-PC tournament tree.
    //    The issuing warp stays unschedulable until served, so each source
    //    has at most NUM_WARPS events outstanding; per-source LUTRAM FIFOs
    //    of that depth absorb collisions (a live event is served directly
    //    when its class is granted and its FIFO is empty).
    localparam ITS_NBAR      = `VX_CFG_ITS_NUM_BARRIERS;
    localparam ITS_ROW_W     = `VX_CFG_NUM_THREADS * PC_BITS;
    localparam ITS_BROW_W    = 2 * ITS_NBAR * `VX_CFG_NUM_THREADS;
    localparam ITS_EV_BR_W   = NW_WIDTH + 2 + 2 * `VX_CFG_NUM_THREADS + ITS_ROW_W + PC_BITS;
    localparam ITS_EV_WC_W   = NW_WIDTH + 2 + ITS_BAR_IDW + 2 * `VX_CFG_NUM_THREADS + PC_BITS;
    localparam ITS_EV_INIT_W = NW_WIDTH + `VX_CFG_NUM_THREADS + PC_BITS;

    reg [`VX_CFG_NUM_WARPS-1:0][PC_BITS-1:0] grp_pc, grp_pc_n;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] grp_mask, grp_mask_n;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] amask, amask_n;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] yielded, yielded_n;
    reg [`VX_CFG_NUM_WARPS-1:0] grp_stale, grp_stale_n;
    reg [`VX_CFG_NUM_WARPS-1:0] ws_pending, ws_pending_n;

    // regroup-engine service outputs (driven in the engine section below)
    logic                           srv_valid;
    logic [NW_WIDTH-1:0]            srv_wid;
    logic [`VX_CFG_NUM_THREADS-1:0] srv_grp_mask;
    logic [PC_BITS-1:0]             srv_grp_pc;
    logic [`VX_CFG_NUM_THREADS-1:0] srv_amask;
    logic [`VX_CFG_NUM_THREADS-1:0] srv_yielded;
    logic                           srv_warp_done;

    `STATIC_ASSERT(`VX_CFG_NUM_ALU_BLOCKS == 1, ("NV_ITS requires a single ALU block"))
`ifdef VX_CFG_EXT_C_ENABLE
    `STATIC_ASSERT(0, ("NV_ITS is incompatible with the C extension"))
`endif
`endif // VX_CFG_DIVERGE_TYPE_NV_ITS
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

`ifdef VX_CFG_DIVERGE_TYPE_SCS
    // SCS parked-split pool: per-warp round-robin FIFO {tmask,pc} in BRAM.
    // 1W1R, addressed by {wid, slot}; registered read gives the 1-cycle pop.
    VX_dp_ram #(
        .DATAW     (CS_W),
        .SIZE      (`VX_CFG_NUM_WARPS * (1 << CS_SLOTW)),   // covers the full {wid,slot} address range
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
`elsif VX_CFG_DIVERGE_TYPE_NV_ITS
    // Warp retirement: threads exit per-group, so the warp is done only when
    // a TMC exit empties the alive set (reported by the regroup engine).
    wire cta_warp_done = srv_valid && srv_warp_done;
`else
    // Warp retirement: TMC with tmask==0 permanently deactivates the warp
    wire cta_warp_done = warp_ctl_if.tmc_valid && (warp_ctl_if.tmc.tmask == 0);
`endif // VX_CFG_DIVERGE_TYPE_SCS

    VX_cta_dispatch #(
        .INSTANCE_ID (`SFORMATF(("%s-cta_dispatch", INSTANCE_ID)))
    ) cta_dispatcher (
        .clk        (clk),
        .reset      (reset),
        .kmu_bus_if (kmu_bus_if),
        .active_warps(active_warps),
        .warp_done  (cta_warp_done),
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
        .warp_done_wid(srv_wid),
`else
        .warp_done_wid(warp_ctl_if.wid),
`endif
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
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0] branch_taken_mask;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0] branch_tmask;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] branch_dest_its;
    wire [`VX_CFG_NUM_ALU_BLOCKS-1:0][PC_BITS-1:0]              branch_ntaken_pc;
`endif
    for (genvar i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin : g_branch_init
        assign branch_valid[i]      = branch_ctl_if[i].valid;
        assign branch_wid[i]        = branch_ctl_if[i].wid;
        assign branch_taken[i]      = branch_ctl_if[i].taken;
        assign branch_dest[i]       = branch_ctl_if[i].dest;
        assign branch_is_trap[i]    = branch_ctl_if[i].is_trap;
        assign branch_is_mret[i]    = branch_ctl_if[i].is_mret;
        assign branch_trap_cause[i] = branch_ctl_if[i].trap_cause;
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
        assign branch_taken_mask[i] = branch_ctl_if[i].taken_mask;
        assign branch_tmask[i]      = branch_ctl_if[i].tmask;
        assign branch_dest_its[i]   = branch_ctl_if[i].dest_its;
        assign branch_ntaken_pc[i]  = branch_ctl_if[i].ntaken_pc;
`endif
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
`ifdef VX_CFG_DIVERGE_TYPE_SCS
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
`endif // VX_CFG_DIVERGE_TYPE_SCS
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
        grp_pc_n     = grp_pc;
        grp_mask_n   = grp_mask;
        amask_n      = amask;
        yielded_n    = yielded;
        grp_stale_n  = grp_stale;
        ws_pending_n = ws_pending;
`endif

        // dispatch warps
        if (cta_fire) begin
            active_warps_n[cta_wid] = 1;
            // Reusing a warp for the next CTA skips the one-time prologue and rewinds to the
            // kernel's per-CTA dispatch window: a fixed 20-byte (5-instruction) sequence that
            // reloads the entry pointer and kargs before re-calling.
            warp_pcs_n[cta_wid] = cta_init ? cta_PC : (warp_pcs[cta_wid] - from_fullPC(`VX_CFG_XLEN'(20)));
            thread_masks_n[cta_wid] = cta_tmask;
`ifdef VX_CFG_DIVERGE_TYPE_SCS
            // SCS: reset per-warp split state for the (re)dispatched CTA.
            cs_pend_n[cta_wid]   = 0;
            cs_cnt_n[cta_wid]    = '0;
            cs_head_n[cta_wid]   = '0;
            cs_done_n[cta_wid]   = '0;
            cs_inpool_n[cta_wid] = '0;
`endif
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
            // ITS: hold the warp unschedulable until its init event has
            // cleared the PC/barrier rows through the regroup engine.
            grp_stale_n[cta_wid] = 1;
`endif
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
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
                    grp_stale_n[i]  = 1;
                    ws_pending_n[i] = 1;
`endif
                end
            end
            stalled_warps_n[wspawn_wid] = 0; // unlock warp
        end

`ifdef VX_CFG_DIVERGE_TYPE_SCS
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
`elsif VX_CFG_DIVERGE_TYPE_NV_ITS
        // ITS: TMC, bar_add/bar_wait and yield are regroup events applied by
        // the engine below; the warp stays parked until its event is served.
        if (warp_ctl_if.tmc_valid || warp_ctl_if.its.valid) begin
            grp_stale_n[warp_ctl_if.wid] = 1;
        end

        // ITS: JOIN is an architectural no-op and the split_join unit is
        // generated away — unlock the warp directly.
        if (warp_ctl_if.sjoin_valid) begin
            stalled_warps_n[warp_ctl_if.wid] = 0;
        end
`else
        // TMC handling
        if (warp_ctl_if.tmc_valid) begin
            active_warps_n[warp_ctl_if.wid]  = (warp_ctl_if.tmc.tmask != 0);
            thread_masks_n[warp_ctl_if.wid]  = warp_ctl_if.tmc.tmask;
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end
`endif // VX_CFG_DIVERGE_TYPE_SCS

`ifdef VX_CFG_DIVERGE_TYPE_SCS
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
        if (scs_enabled && warp_ctl_if.pred_park_valid) begin
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
        if (scs_enabled && warp_ctl_if.pred_restore_valid) begin
            if (cs_pend[warp_ctl_if.wid] && (cs_ppc[warp_ctl_if.wid] == warp_pcs[warp_ctl_if.wid])) begin
                thread_masks_n[warp_ctl_if.wid] = (thread_masks[warp_ctl_if.wid] | cs_ptmask[warp_ctl_if.wid]) & ~cs_done[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = ((thread_masks[warp_ctl_if.wid] | cs_ptmask[warp_ctl_if.wid]) & ~cs_done[warp_ctl_if.wid]) != 0;
                cs_pend_n[warp_ctl_if.wid]      = 0;
            end else begin
                thread_masks_n[warp_ctl_if.wid] = thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid];
                active_warps_n[warp_ctl_if.wid] = (thread_masks[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]) != 0;
            end
        end
`endif // VX_CFG_DIVERGE_TYPE_SCS

        // split handling (no-op under ITS: divergence is per-thread PC)
        if (warp_ctl_if.split_valid) begin
`ifndef VX_CFG_DIVERGE_TYPE_NV_ITS
            if (warp_ctl_if.split.is_dvg) begin
                thread_masks_n[warp_ctl_if.wid] = warp_ctl_if.split.then_tmask;
            end
`endif
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end

        // join handling (no-op under ITS: reconvergence is bar_wait / min-PC grouping)
        if (join_valid) begin
`ifndef VX_CFG_DIVERGE_TYPE_NV_ITS
            if (join_is_dvg) begin
                if (join_is_else) begin
                    warp_pcs_n[join_wid] = join_pc;
                end
                thread_masks_n[join_wid] = join_tmask;
            end
`endif
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

`ifdef VX_CFG_DIVERGE_TYPE_SCS
        // SCS: vx_yield — defer the running (spinning) split and run the next
        // runnable one so a lock holder makes progress while spinners wait.
        // Pending acquirers run immediately; else rotate to the oldest pooled
        // split. The pool is 1W1R, so pushing the current split at the tail and
        // popping the head happen in the same cycle (distinct slots). No-op when
        // nothing else is runnable.
        if (scs_enabled && warp_ctl_if.yield_valid) begin
            // Installing pending requires pushing the current split to the pool
            // tail (net +1 occupancy), so it is only taken with free capacity
            // (case-study K-sweep: CS_DEPTH can be << NUM_THREADS). When the pool
            // is full, cs_pend/cs_ptmask/cs_ppc are left untouched — never
            // dropped — and picked up later either by a subsequent yield once a
            // pop frees a slot, or directly by this warp's own kernel-exit path,
            // which installs pending without touching the pool at all.
            if (cs_pend[warp_ctl_if.wid]
             && (cs_ptmask[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]) != 0
             && (cs_cnt[warp_ctl_if.wid] < CS_CW'(CS_DEPTH))) begin
                // defer current to the tail, run pending acquirers directly
                cs_pend_n[warp_ctl_if.wid] = 0;
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

        // A/B ipdom mode: yield is a plain warp unlock (no split rotation).
        if (!scs_enabled && warp_ctl_if.yield_valid) begin
            stalled_warps_n[warp_ctl_if.wid] = 0;
        end
`endif // VX_CFG_DIVERGE_TYPE_SCS

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
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
                // the per-thread PC writes and the warp unlock happen when
                // the regroup engine serves this resolution.
                grp_stale_n[branch_wid[i]] = 1;
`else
                stalled_warps_n[branch_wid[i]] = 0; // unlock warp
`endif
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
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
            // ITS: the issued group advances to the fallthrough PC.
            grp_pc_n[schedule_if.data.wid] = schedule_if.data.PC + from_fullPC(`VX_CFG_XLEN'(4));
`endif
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

`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
        // ITS regroup service: install the recomputed group, release the warp,
        // and retire it once a kernel-exit TMC has emptied the alive set.
        if (srv_valid) begin
            grp_pc_n[srv_wid]        = srv_grp_pc;
            grp_mask_n[srv_wid]      = srv_grp_mask;
            amask_n[srv_wid]         = srv_amask;
            yielded_n[srv_wid]       = srv_yielded;
            grp_stale_n[srv_wid]     = 0;
            ws_pending_n[srv_wid]    = 0;
            thread_masks_n[srv_wid]  = srv_grp_mask;
            stalled_warps_n[srv_wid] = 0;
            if (srv_warp_done) begin
                active_warps_n[srv_wid] = 0;
            end else if (srv_amask != 0) begin
                active_warps_n[srv_wid] = 1;
            end
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
`ifdef VX_CFG_DIVERGE_TYPE_SCS
            cs_pend         <= '0;
            cs_cnt          <= '0;
            cs_head         <= '0;
            cs_done         <= '0;
            cs_inpool       <= '0;
            cs_pop_valid_r  <= '0;
`endif
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
            grp_pc          <= '0;
            grp_mask        <= '0;
            amask           <= '0;
            yielded         <= '0;
            grp_stale       <= '0;
            ws_pending      <= '0;
`endif
        end else begin
            active_warps   <= active_warps_n;
            stalled_warps  <= stalled_warps_n;
            thread_masks   <= thread_masks_n;
            warp_pcs       <= warp_pcs_n;
`ifdef VX_CFG_DIVERGE_TYPE_SCS
            cs_pend        <= cs_pend_n;
            cs_ptmask      <= cs_ptmask_n;
            cs_ppc         <= cs_ppc_n;
            cs_head        <= cs_head_n;
            cs_cnt         <= cs_cnt_n;
            cs_done        <= cs_done_n;
            cs_inpool      <= cs_inpool_n;
            cs_pop_valid_r <= cs_pop_set;
            cs_pop_wid_r   <= cs_pop_wid;
`endif
`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
            grp_pc         <= grp_pc_n;
            grp_mask       <= grp_mask_n;
            amask          <= amask_n;
            yielded        <= yielded_n;
            grp_stale      <= grp_stale_n;
            ws_pending     <= ws_pending_n;
`endif
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

`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    // ITS carries no IPDOM stack: split/join are architectural no-ops, so the
    // divergence-stack storage is generated away entirely (fair area account).
    assign join_valid   = 1'b0;
    assign join_is_dvg  = 1'b0;
    assign join_is_else = 1'b0;
    assign join_wid     = '0;
    assign join_tmask   = '0;
    assign join_pc      = '0;
    assign warp_ctl_if.dvstack_ptr = '0;
    `UNUSED_VAR (warp_ctl_if.split)
    `UNUSED_VAR (warp_ctl_if.sjoin)
    `UNUSED_VAR (warp_ctl_if.dvstack_wid)
`else
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
`endif // !VX_CFG_DIVERGE_TYPE_NV_ITS

    // schedule the next ready warp

`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    // ITS regroup engine (see the state-declaration comment for the design).

    // event capture: branch resolution
    wire br_ev_live = branch_valid[0];
    wire [ITS_EV_BR_W-1:0] br_ev_live_data = {
        branch_wid[0], branch_is_trap[0], branch_is_mret[0],
        branch_taken_mask[0], branch_tmask[0], branch_dest_its[0], branch_ntaken_pc[0]};

    // event capture: wctl (bar_add / bar_wait / yield / tmc)
    wire wc_ev_live = warp_ctl_if.tmc_valid || warp_ctl_if.its.valid;
    wire [1:0] wc_ev_live_op = warp_ctl_if.tmc_valid ? 2'd3 :
                               (warp_ctl_if.its.is_yield ? 2'd2 :
                               (warp_ctl_if.its.is_wait ? 2'd1 : 2'd0));
    wire [ITS_EV_WC_W-1:0] wc_ev_live_data = {
        warp_ctl_if.wid, wc_ev_live_op, warp_ctl_if.its.bid,
        warp_ctl_if.its.tmask, warp_ctl_if.tmc.tmask, warp_ctl_if.its_pc};

    // event capture: warp init (CTA dispatch, or one legacy-wspawn warp/cycle)
    wire [PC_BITS-1:0] cta_init_pc = cta_init ? cta_PC : (warp_pcs[cta_wid] - from_fullPC(`VX_CFG_XLEN'(20)));
    wire ws_ev_live = (ws_pending != 0) && !cta_fire;
    wire [NW_WIDTH-1:0] ws_ev_wid;
    VX_priority_encoder #(
        .N (`VX_CFG_NUM_WARPS)
    ) ws_enc (
        .data_in   (ws_pending),
        .index_out (ws_ev_wid),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );
    wire init_ev_live = cta_fire || ws_ev_live;
    wire [ITS_EV_INIT_W-1:0] init_ev_live_data = cta_fire
        ? {cta_wid, cta_tmask, cta_init_pc}
        : {ws_ev_wid, `VX_CFG_NUM_THREADS'(1), wspawn.pc};

    // yield wake: alive threads, empty group, some yielded
    wire [`VX_CFG_NUM_WARPS-1:0] wake_eligible;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_wake_eligible
        assign wake_eligible[w] = active_warps[w] && !stalled_warps[w] && !grp_stale[w]
                               && (grp_mask[w] == 0) && ((amask[w] & yielded[w]) != 0);
    end
    wire wake_any;
    wire [NW_WIDTH-1:0] wake_wid;
    VX_priority_encoder #(
        .N (`VX_CFG_NUM_WARPS)
    ) wake_enc (
        .data_in   (wake_eligible),
        .index_out (wake_wid),
        .valid_out (wake_any),
        `UNUSED_PIN (onehot_out)
    );

    // per-class FIFOs; a class serves its FIFO head before its live event
    wire br_q_pop, wc_q_pop, in_q_pop;
    wire br_q_push, wc_q_push, in_q_push;
    wire br_q_empty, wc_q_empty, in_q_empty;
    wire br_q_full, wc_q_full, in_q_full;
    wire [ITS_EV_BR_W-1:0]   br_q_data;
    wire [ITS_EV_WC_W-1:0]   wc_q_data;
    wire [ITS_EV_INIT_W-1:0] in_q_data;

    VX_fifo_queue #(
        .DATAW  (ITS_EV_BR_W),
        .DEPTH  (`VX_CFG_NUM_WARPS),
        .LUTRAM (1)
    ) br_ev_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (br_q_push),
        .pop      (br_q_pop),
        .data_in  (br_ev_live_data),
        .data_out (br_q_data),
        .empty    (br_q_empty),
        .full     (br_q_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    VX_fifo_queue #(
        .DATAW  (ITS_EV_WC_W),
        .DEPTH  (`VX_CFG_NUM_WARPS),
        .LUTRAM (1)
    ) wc_ev_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (wc_q_push),
        .pop      (wc_q_pop),
        .data_in  (wc_ev_live_data),
        .data_out (wc_q_data),
        .empty    (wc_q_empty),
        .full     (wc_q_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    VX_fifo_queue #(
        .DATAW  (ITS_EV_INIT_W),
        .DEPTH  (`VX_CFG_NUM_WARPS),
        .LUTRAM (1)
    ) in_ev_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (in_q_push),
        .pop      (in_q_pop),
        .data_in  (init_ev_live_data),
        .data_out (in_q_data),
        .empty    (in_q_empty),
        .full     (in_q_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire [3:0] ev_req;
    assign ev_req[0] = !br_q_empty || br_ev_live;
    assign ev_req[1] = !wc_q_empty || wc_ev_live;
    assign ev_req[2] = !in_q_empty || init_ev_live;
    assign ev_req[3] = wake_any;

    wire [3:0] ev_grant;
    wire ev_grant_valid;
    VX_rr_arbiter #(
        .NUM_REQS (4)
    ) ev_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (ev_req),
        `UNUSED_PIN (grant_index),
        .grant_onehot (ev_grant),
        .grant_valid  (ev_grant_valid),
        .grant_ready  (ev_grant_valid)
    );

    wire br_srv = ev_grant_valid && ev_grant[0];
    wire wc_srv = ev_grant_valid && ev_grant[1];
    wire in_srv = ev_grant_valid && ev_grant[2];
    wire wk_srv = ev_grant_valid && ev_grant[3];

    assign br_q_pop  = br_srv && !br_q_empty;
    assign wc_q_pop  = wc_srv && !wc_q_empty;
    assign in_q_pop  = in_srv && !in_q_empty;
    assign br_q_push = br_ev_live && !(br_srv && br_q_empty);
    assign wc_q_push = wc_ev_live && !(wc_srv && wc_q_empty);
    assign in_q_push = cta_fire && !(in_srv && in_q_empty && cta_fire);
    `RUNTIME_ASSERT(!(br_q_push && br_q_full), ("%t: %s ITS branch-event queue overflow", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(!(wc_q_push && wc_q_full), ("%t: %s ITS wctl-event queue overflow", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(!(in_q_push && in_q_full), ("%t: %s ITS init-event queue overflow", $time, INSTANCE_ID))

    // selected event fields
    wire [ITS_EV_BR_W-1:0]   br_ev_data = br_q_empty ? br_ev_live_data : br_q_data;
    wire [ITS_EV_WC_W-1:0]   wc_ev_data = wc_q_empty ? wc_ev_live_data : wc_q_data;
    wire [ITS_EV_INIT_W-1:0] in_ev_data = in_q_empty ? init_ev_live_data : in_q_data;

    wire [NW_WIDTH-1:0] br_ev_wid;
    wire br_ev_is_trap, br_ev_is_mret;
    wire [`VX_CFG_NUM_THREADS-1:0] br_ev_taken, br_ev_group;
    wire [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] br_ev_dests;
    wire [PC_BITS-1:0] br_ev_ntaken;
    assign {br_ev_wid, br_ev_is_trap, br_ev_is_mret, br_ev_taken, br_ev_group, br_ev_dests, br_ev_ntaken} = br_ev_data;

    wire [NW_WIDTH-1:0] wc_ev_wid;
    wire [1:0] wc_ev_op; // 0 = bar_add, 1 = bar_wait, 2 = yield, 3 = tmc
    wire [ITS_BAR_IDW-1:0] wc_ev_bid;
    wire [`VX_CFG_NUM_THREADS-1:0] wc_ev_group, wc_ev_tmc_tmask;
    wire [PC_BITS-1:0] wc_ev_pc;
    assign {wc_ev_wid, wc_ev_op, wc_ev_bid, wc_ev_group, wc_ev_tmc_tmask, wc_ev_pc} = wc_ev_data;

    wire [NW_WIDTH-1:0] in_ev_wid;
    wire [`VX_CFG_NUM_THREADS-1:0] in_ev_tmask;
    wire [PC_BITS-1:0] in_ev_pc;
    assign {in_ev_wid, in_ev_tmask, in_ev_pc} = in_ev_data;

    // row stores: per-thread PCs and the {participate, arrived} barrier masks.
    // The service datapath is a 2-stage pipeline to keep 300 MHz: stage 1
    // (st1_*) reads the rows, applies the event and re-evaluates barrier
    // release; stage 2 (s1_* registered) runs the min-PC tree and installs the
    // group. The serviced warp is schedule-stalled throughout, so the extra
    // cycle is hidden behind the other warps.
    logic                                        st1_valid;
    logic [NW_WIDTH-1:0]                          st1_wid;
    logic [`VX_CFG_NUM_THREADS-1:0]              st1_wren;
    logic [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] st1_wpcs;
    logic                                        st1_bar_we;
    logic [ITS_NBAR-1:0][`VX_CFG_NUM_THREADS-1:0] part_new, arr_new;
    wire  [ITS_NBAR-1:0][`VX_CFG_NUM_THREADS-1:0] part_row, arr_row;
    wire  [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] tpc_row;

    VX_dp_ram #(
        .DATAW    (ITS_ROW_W),
        .SIZE     (`VX_CFG_NUM_WARPS),
        .WRENW    (`VX_CFG_NUM_THREADS),
        .OUT_REG  (0),
        .LUTRAM   (1),
        .RDW_MODE ("R")
    ) tpc_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (st1_valid && (st1_wren != 0)),
        .wren  (st1_wren),
        .waddr (st1_wid),
        .wdata (st1_wpcs),
        .raddr (st1_wid),
        .rdata (tpc_row)
    );
    VX_dp_ram #(
        .DATAW    (ITS_BROW_W),
        .SIZE     (`VX_CFG_NUM_WARPS),
        .OUT_REG  (0),
        .LUTRAM   (1),
        .RDW_MODE ("R")
    ) bar_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (st1_valid && st1_bar_we),
        .wren  (1'b1),
        .waddr (st1_wid),
        .wdata ({part_new, arr_new}),
        .raddr (st1_wid),
        .rdata ({part_row, arr_row})
    );

    localparam ITS_MIN_LVLS = `CLOG2(`VX_CFG_NUM_THREADS);

    // stage-1 combinational: pick the event, read the rows, apply the event and
    // re-evaluate barrier release. Produces the post-event per-thread PCs and
    // runnable set for the tree, plus the row-store writes (committed this cycle).
    logic                                        st1_keep_grp;
    logic [`VX_CFG_NUM_THREADS-1:0]              st1_amask, st1_yielded;
    logic [`VX_CFG_NUM_THREADS-1:0]              st1_runnable;
    logic [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] st1_pcs;
    logic                                        st1_warp_done;
    logic [PC_BITS-1:0]                          st1_keep_pc;
    logic [`VX_CFG_NUM_THREADS-1:0]              st1_keep_mask;
    always @(*) begin
        logic [`VX_CFG_NUM_THREADS-1:0] blocked;

        st1_valid     = ev_grant_valid;
        st1_wid       = br_ev_wid;
        st1_wren      = '0;
        st1_wpcs      = '0;
        st1_bar_we    = 1'b0;
        st1_warp_done = 1'b0;
        st1_keep_grp  = 1'b0;

        if (wc_srv) begin
            st1_wid = wc_ev_wid;
        end else if (in_srv) begin
            st1_wid = in_ev_wid;
        end else if (wk_srv) begin
            st1_wid = wake_wid;
        end

        st1_amask   = amask[st1_wid];
        st1_yielded = yielded[st1_wid];
        part_new    = part_row;
        arr_new     = arr_row;

        if (br_srv) begin
            st1_wren = br_ev_group;
            for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                if (br_ev_is_trap) begin
                    st1_wpcs[t] = from_fullPC(mtvec_r[st1_wid] & ~`VX_CFG_XLEN'(3));
                end else if (br_ev_is_mret) begin
                    st1_wpcs[t] = from_fullPC(mepc_r[st1_wid]);
                end else begin
                    st1_wpcs[t] = br_ev_taken[t] ? br_ev_dests[t] : br_ev_ntaken;
                end
            end
        end else if (wc_srv) begin
            case (wc_ev_op)
                2'd0: begin // bar_add: the group registers as participants.
                    // No regroup: the running group's row entries are stale
                    // (their truth is grp_pc), and nothing became runnable.
                    part_new[wc_ev_bid] = part_row[wc_ev_bid] | wc_ev_group;
                    st1_bar_we   = 1'b1;
                    st1_keep_grp = 1'b1;
                end
                2'd1: begin // bar_wait: masked arrival; empty barrier passes
                    st1_wren = wc_ev_group;
                    for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                        st1_wpcs[t] = wc_ev_pc;
                    end
                    arr_new[wc_ev_bid] = arr_row[wc_ev_bid] | (wc_ev_group & part_row[wc_ev_bid]);
                    st1_bar_we = 1'b1;
                end
                2'd2: begin // yield: the group enters the Yielded state
                    st1_wren = wc_ev_group;
                    for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                        st1_wpcs[t] = wc_ev_pc;
                    end
                    st1_yielded = yielded[st1_wid] | wc_ev_group;
                    st1_bar_we = 1'b1;
                end
                default: begin // tmc: mask set, or per-group kernel exit
                    if (wc_ev_tmc_tmask == 0) begin
                        st1_amask = amask[st1_wid] & ~wc_ev_group;
                        st1_warp_done = (st1_amask == 0);
                    end else begin
                        st1_amask = wc_ev_tmc_tmask;
                        st1_wren  = wc_ev_tmc_tmask;
                        for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                            st1_wpcs[t] = wc_ev_pc;
                        end
                    end
                    st1_yielded = yielded[st1_wid] & st1_amask;
                    for (integer b = 0; b < ITS_NBAR; ++b) begin
                        part_new[b] = part_row[b] & st1_amask;
                        arr_new[b]  = arr_row[b] & st1_amask;
                    end
                    st1_bar_we = 1'b1;
                end
            endcase
            // release every barrier whose still-expected participants (alive,
            // not yielded) have all arrived; clearing both masks makes the bid
            // reusable. Yielded parkers must not hold back reconvergence.
            for (integer b = 0; b < ITS_NBAR; ++b) begin
                if ((part_new[b] != 0) && (arr_new[b] != 0)
                 && ((part_new[b] & ~st1_yielded & ~arr_new[b]) == 0)) begin
                    part_new[b] = '0;
                    arr_new[b]  = '0;
                end
            end
        end else if (in_srv) begin
            // init: all dispatched threads start alive at the entry PC with
            // clean barrier state
            st1_amask   = in_ev_tmask;
            st1_yielded = '0;
            st1_wren    = in_ev_tmask;
            for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                st1_wpcs[t] = in_ev_pc;
            end
            part_new   = '0;
            arr_new    = '0;
            st1_bar_we = 1'b1;
        end else if (wk_srv) begin
            // wake: nothing else in the warp can run - clear the yielded set
            st1_yielded = '0;
        end

        // post-event runnable set + per-thread PCs (event-written PCs bypass the
        // row store, which returns pre-write data). The min-PC tree over these
        // runs in stage 2.
        blocked = '0;
        for (integer b = 0; b < ITS_NBAR; ++b) begin
            blocked |= arr_new[b];
        end
        st1_runnable = st1_amask & ~blocked & ~st1_yielded;
        // Wake-when-nothing-else-runnable, applied in-service so the just-
        // yielded group's fresh resume PC comes from st1_wpcs (row bypass)
        // rather than a next-cycle row read that races the yield's RAM write.
        if ((st1_runnable == 0) && ((st1_amask & ~blocked & st1_yielded) != 0)) begin
            st1_yielded  = st1_yielded & ~(st1_amask & ~blocked);
            st1_runnable = st1_amask & ~blocked & ~st1_yielded;
        end
        for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
            st1_pcs[t] = st1_wren[t] ? st1_wpcs[t] : tpc_row[t];
        end
        st1_keep_pc   = grp_pc[st1_wid];
        st1_keep_mask = grp_mask[st1_wid];
    end

    // stage-1 → stage-2 pipeline register
    reg                                        s1_valid;
    reg [NW_WIDTH-1:0]                          s1_wid;
    reg [`VX_CFG_NUM_THREADS-1:0]              s1_amask, s1_yielded;
    reg [`VX_CFG_NUM_THREADS-1:0]              s1_runnable;
    reg [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] s1_pcs;
    reg                                        s1_warp_done, s1_keep_grp;
    reg [PC_BITS-1:0]                          s1_keep_pc;
    reg [`VX_CFG_NUM_THREADS-1:0]              s1_keep_mask;
    always @(posedge clk) begin
        if (reset) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid     <= st1_valid;
            s1_wid       <= st1_wid;
            s1_amask     <= st1_amask;
            s1_yielded   <= st1_yielded;
            s1_runnable  <= st1_runnable;
            s1_pcs       <= st1_pcs;
            s1_warp_done <= st1_warp_done;
            s1_keep_grp  <= st1_keep_grp;
            s1_keep_pc   <= st1_keep_pc;
            s1_keep_mask <= st1_keep_mask;
        end
    end

    // stage-2 combinational: min-PC tournament tree + group install signals.
    always @(*) begin
        logic [`VX_CFG_NUM_THREADS-1:0][PC_BITS-1:0] lvl;

        srv_valid     = s1_valid;
        srv_wid       = s1_wid;
        srv_amask     = s1_amask;
        srv_yielded   = s1_yielded;
        srv_warp_done = s1_warp_done;

        for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
            lvl[t] = s1_runnable[t] ? s1_pcs[t] : {PC_BITS{1'b1}};
        end
        for (integer l = 0; l < ITS_MIN_LVLS; ++l) begin
            for (integer i = 0; i < (`VX_CFG_NUM_THREADS >> (l + 1)); ++i) begin
                lvl[i] = (lvl[2*i] < lvl[2*i+1]) ? lvl[2*i] : lvl[2*i+1];
            end
        end
        srv_grp_pc = lvl[0];
        for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
            srv_grp_mask[t] = s1_runnable[t] && (s1_pcs[t] == srv_grp_pc);
        end
        if (s1_keep_grp) begin
            srv_grp_pc   = s1_keep_pc;
            srv_grp_mask = s1_keep_mask;
        end
    end

    // a warp is schedulable once its group cache is valid and non-empty
    wire [`VX_CFG_NUM_WARPS-1:0] grp_any;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_grp_any
        assign grp_any[w] = (grp_mask[w] != 0);
    end
    wire [`VX_CFG_NUM_WARPS-1:0] ready_warps = active_warps & ~stalled_warps & ~grp_stale & grp_any;
`else
    wire [`VX_CFG_NUM_WARPS-1:0] ready_warps = active_warps & ~stalled_warps;
`endif

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

`ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    `UNUSED_VAR (schedule_data)
    assign schedule_tmask = grp_mask[schedule_wid];
    assign schedule_pc    = grp_pc[schedule_wid];
`else
    assign {schedule_tmask, schedule_pc} = {
        schedule_data[schedule_wid][(`VX_CFG_NUM_THREADS + PC_BITS)-1:(`VX_CFG_NUM_THREADS + PC_BITS)-4],
        schedule_data[schedule_wid][(`VX_CFG_NUM_THREADS + PC_BITS)-5:0]
    };
`endif

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

`ifdef VX_CFG_DIVERGE_TYPE_SCS
`ifdef THREADSPLIT_EVAL
    longint unsigned eval_issued;
    longint unsigned eval_lanes;
    longint unsigned eval_yields;
    longint unsigned eval_switches;
    longint unsigned eval_peak;
    longint unsigned eval_peak_next;
    // Case-study A/B instrumentation: per-warp-cycle "live" context count is the
    // same quantity eval_peak already maximizes (running + pending + pooled);
    // here every warp-cycle with >=1 live context is also binned into a
    // histogram (fig. A1) and "runnable" (pending + pooled, i.e. everything
    // *other* than the currently running split) is separately summed/maxed.
    longint unsigned eval_live_cycles;    // sum of warp-cycles with >=1 live context
    longint unsigned eval_hist1, eval_hist2, eval_hist3, eval_hist4, eval_hist5p;
    longint unsigned eval_runnable_sum;
    longint unsigned eval_runnable_peak, eval_runnable_peak_next;
    wire eval_tmc_exit = warp_ctl_if.tmc_valid && warp_ctl_if.tmc.tmask == 0;
    wire eval_pending_switch = scs_enabled && (warp_ctl_if.yield_valid || eval_tmc_exit)
                            && cs_pend[warp_ctl_if.wid]
                            && ((cs_ptmask[warp_ctl_if.wid] & ~cs_done[warp_ctl_if.wid]
                               & ~(eval_tmc_exit ? thread_masks[warp_ctl_if.wid] : '0)) != 0);
    initial begin
        eval_issued = 0;
        eval_lanes = 0;
        eval_yields = 0;
        eval_switches = 0;
        eval_peak = 0;
        eval_live_cycles = 0;
        eval_hist1 = 0;
        eval_hist2 = 0;
        eval_hist3 = 0;
        eval_hist4 = 0;
        eval_hist5p = 0;
        eval_runnable_sum = 0;
        eval_runnable_peak = 0;
    end
    // Per-cycle deltas, summed combinationally across all warps. Each accumulator
    // below is written by exactly one nonblocking assignment per clock edge; do
    // NOT fold the per-warp loop into the posedge block directly; NUM_WARPS
    // nonblocking writes to the same register in one always block all sample the
    // same pre-edge value; only the last write survives, silently undercounting
    // whenever >=2 warps hit the same bin in the same cycle.
    longint unsigned eval_delta_live_cycles;
    longint unsigned eval_delta_runnable_sum;
    longint unsigned eval_delta_hist1, eval_delta_hist2, eval_delta_hist3, eval_delta_hist4, eval_delta_hist5p;
    always @(*) begin
        eval_peak_next = eval_peak;
        eval_runnable_peak_next = eval_runnable_peak;
        eval_delta_live_cycles = 0;
        eval_delta_runnable_sum = 0;
        eval_delta_hist1 = 0;
        eval_delta_hist2 = 0;
        eval_delta_hist3 = 0;
        eval_delta_hist4 = 0;
        eval_delta_hist5p = 0;
        for (integer w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
            automatic longint unsigned live_w = 64'(active_warps[w] && thread_masks[w] != 0)
                                               + 64'(cs_pend[w]) + 64'(cs_cnt[w]);
            automatic longint unsigned runnable_w = 64'(cs_pend[w]) + 64'(cs_cnt[w]);
            if (scs_enabled && live_w > eval_peak_next) begin
                eval_peak_next = live_w;
            end
            if (scs_enabled && runnable_w > eval_runnable_peak_next) begin
                eval_runnable_peak_next = runnable_w;
            end
            if (scs_enabled && live_w != 0) begin
                eval_delta_live_cycles = eval_delta_live_cycles + 1;
                eval_delta_runnable_sum = eval_delta_runnable_sum + runnable_w;
                case (live_w)
                    64'd1:   eval_delta_hist1  = eval_delta_hist1  + 1;
                    64'd2:   eval_delta_hist2  = eval_delta_hist2  + 1;
                    64'd3:   eval_delta_hist3  = eval_delta_hist3  + 1;
                    64'd4:   eval_delta_hist4  = eval_delta_hist4  + 1;
                    default: eval_delta_hist5p = eval_delta_hist5p + 1;
                endcase
            end
        end
    end
    always @(posedge clk) begin
        if (!reset) begin
            if (schedule_if_fire) begin
                eval_issued <= eval_issued + 1;
                eval_lanes <= eval_lanes + 64'($countones(schedule_if.data.tmask));
            end
            if (warp_ctl_if.yield_valid) begin
                eval_yields <= eval_yields + 1;
            end
            eval_switches <= eval_switches + 64'(cs_pop_valid_r) + 64'(eval_pending_switch);
            eval_peak <= eval_peak_next;
            eval_runnable_peak <= eval_runnable_peak_next;
            eval_live_cycles  <= eval_live_cycles  + eval_delta_live_cycles;
            eval_runnable_sum <= eval_runnable_sum + eval_delta_runnable_sum;
            eval_hist1  <= eval_hist1  + eval_delta_hist1;
            eval_hist2  <= eval_hist2  + eval_delta_hist2;
            eval_hist3  <= eval_hist3  + eval_delta_hist3;
            eval_hist4  <= eval_hist4  + eval_delta_hist4;
            eval_hist5p <= eval_hist5p + eval_delta_hist5p;
        end
    end
    final begin
        $display("EVAL: core=%0d issued=%0d active_lanes=%0d yields=%0d switches=%0d watchdog_switches=0 peak_contexts=%0d live_cycles=%0d hist1=%0d hist2=%0d hist3=%0d hist4=%0d hist5p=%0d runnable_sum=%0d runnable_peak=%0d",
                 CORE_ID, eval_issued, eval_lanes, eval_yields, eval_switches, eval_peak,
                 eval_live_cycles, eval_hist1, eval_hist2, eval_hist3, eval_hist4, eval_hist5p,
                 eval_runnable_sum, eval_runnable_peak);
    end
`endif // THREADSPLIT_EVAL
`endif // VX_CFG_DIVERGE_TYPE_SCS
endmodule
