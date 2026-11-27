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

// The state of one warp of a frontend slice: active, stalled, thread mask, PC,
// its trap CSRs, its ibuffer occupancy and its in-flight instruction count.
// Events arrive already addressed to the warp (the *_hit inputs), so every
// warp of the slice is the same module; the FU ports are the ones the warp is
// bound to.
module VX_warp_sched_ctx import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // the warp's slice-local index and slice, for tracing
    input wire [FETCH_WFS_W-1:0] lid,
    input wire [FETCH_FSW_W-1:0] slice_id,

    // CTA launch
    input wire              cta_hit,
    input wire              cta_init,
    input wire [PC_BITS-1:0] cta_PC,
    input wire [`VX_CFG_NUM_THREADS-1:0] cta_tmask,
    input wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param,

    // wspawn
    input wire              wspawn_apply,
    input wire              wspawn_sel,
    input wire              wspawn_src_hit,
    input wire [PC_BITS-1:0] wspawn_pc,
    input wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] wspawn_mscratch,

    // barrier release and retirement
    input wire              bar_unlock_valid,
    input wire              bar_unlock_sel,
    input wire              retire_hit,

    input wire [`VX_CFG_XLEN-1:0] mtvec,

    // decode
    input wire              dec_hit,
    input wire              dec_unlock,
    input wire              dec_fs_dirty,
`ifdef VX_CFG_EXT_C_ENABLE
    input wire              dec_is_rvc,
`endif

    // the warp's ALU port
    input sched_branch_t    branch,
    input wire              br_hit,

    // the warp's SFU port
    input sched_wctl_t      wctl,
    input wire              ctl_hit,
    input sched_csr_wr_t    csr_wr,
    input wire              csr_wr_hit,
`ifdef VX_CFG_EXT_RTU_ENABLE
    input wire              rtu_unlock_hit,
`endif

    // the SFU port's split/join stack
    input wire              join_hit,
    input wire              join_is_dvg,
    input wire              join_is_else,
    input wire [`VX_CFG_NUM_THREADS-1:0] join_tmask,
    input wire [PC_BITS-1:0] join_pc,

    // scheduling and pipeline events
    input wire              sched_fire,
    input wire              sched_sel,
    input wire              ibuf_pop,
    input wire              issued,
    input wire              committed,

    // warp state
    output wire             active,
    output wire             stalled,
    output wire [`VX_CFG_NUM_THREADS-1:0] tmask,
    output wire [PC_BITS-1:0] pc,
    output wire             ibuf_full,
    output wire             pending_empty,
    output wire             pending_alm_empty,

    // trap CSRs
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch,
    output wire [`VX_CFG_XLEN-1:0] mstatus,
    output wire [`VX_CFG_XLEN-1:0] mepc,
    output wire [TRAP_CAUSE_BITS-1:0] mcause,
    output wire             fs_off
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({branch.valid, branch.wid})
    `UNUSED_VAR ({wctl.sjoin_valid, wctl.wid, wctl.split.else_tmask, wctl.split.next_pc, wctl.sjoin})
    `UNUSED_VAR (csr_wr.wid)
`ifndef DBG_TRACE_PIPELINE
    `UNUSED_VAR ({lid, slice_id})
`endif

    localparam IBUF_CW = $clog2(`VX_CFG_IBUF_SIZE + 1);

    // M is the only privilege mode, so mstatus keeps MIE/MPIE/FS, and mepc
    // keeps the bits that can be set.
`ifdef VX_CFG_EXT_C_ENABLE
    localparam MEPC_LSB  = 1;
`else
    localparam MEPC_LSB  = 2;
`endif
    // FS resets to Initial so that FP code runs without first writing mstatus.
    // Without F, FS is read-only zero.
`ifdef VX_CFG_EXT_F_ENABLE
    localparam [1:0] FS_RESET = 2'b01;
    localparam [1:0] FS_WMASK = 2'b11;
`else
    localparam [1:0] FS_RESET = 2'b00;
    localparam [1:0] FS_WMASK = 2'b00;
`endif

    function automatic logic [`VX_CFG_XLEN-1:0] mstatus_value(
        input logic       mie,
        input logic       mpie,
        input logic [1:0] fs
    );
        mstatus_value = '0;
        mstatus_value[`VX_CSR_MSTATUS_MIE]      = mie;
        mstatus_value[`VX_CSR_MSTATUS_MPIE]     = mpie;
        mstatus_value[`VX_CSR_MSTATUS_MPP +: 2] = 2'b11;
        mstatus_value[`VX_CSR_MSTATUS_FS +: 2]  = fs;
        mstatus_value[`VX_CFG_XLEN-1]           = (fs == 2'b11); // SD
    endfunction

    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch_r;
    reg mstatus_mie_r, mstatus_mpie_r;
    reg [1:0] mstatus_fs_r;
    reg [`VX_CFG_XLEN-1:MEPC_LSB] mepc_r;
    reg [TRAP_CAUSE_BITS-1:0] mcause_r;

    reg active_r, active_n;
    reg stalled_r, stalled_n;
    reg [`VX_CFG_NUM_THREADS-1:0] tmask_r, tmask_n;
    reg [PC_BITS-1:0] pc_r, pc_n;

    assign mstatus = mstatus_value(mstatus_mie_r, mstatus_mpie_r, mstatus_fs_r);
    assign mepc    = {mepc_r, MEPC_LSB'(0)};
    assign fs_off  = (mstatus_fs_r == 2'b00);

    always @(*) begin
        active_n  = active_r;
        stalled_n = stalled_r;
        tmask_n   = tmask_r;
        pc_n      = pc_r;

        if (cta_hit) begin
            active_n = 1;
            // Reusing a warp for the next CTA skips the one-time prologue and rewinds to the
            // kernel's per-CTA dispatch window: a fixed 20-byte (5-instruction) sequence that
            // reloads the entry pointer and kargs before re-calling.
            pc_n    = cta_init ? cta_PC : (pc_r - from_fullPC(`VX_CFG_XLEN'(20)));
            tmask_n = cta_tmask;
        end

        if (dec_hit && dec_unlock) begin
            stalled_n = 0;
        end

        if (csr_wr.unlock && csr_wr_hit) begin
            stalled_n = 0;
        end

        if (wspawn_apply) begin
            if (wspawn_sel) begin
                active_n = 1;
                if (!wspawn_src_hit) begin
                    tmask_n[0] = 1;
                    pc_n = wspawn_pc;
                end
            end
            if (wspawn_src_hit) begin
                stalled_n = 0;
            end
        end

        if (wctl.tmc_valid && ctl_hit) begin
            active_n  = (wctl.tmc.tmask != 0);
            tmask_n   = wctl.tmc.tmask;
            stalled_n = 0;
        end

        if (wctl.split_valid && ctl_hit) begin
            if (wctl.split.is_dvg) begin
                tmask_n = wctl.split.then_tmask;
            end
            stalled_n = 0;
        end

        if (join_hit) begin
            if (join_is_dvg) begin
                if (join_is_else) begin
                    pc_n = join_pc;
                end
                tmask_n = join_tmask;
            end
            stalled_n = 0;
        end

        if (bar_unlock_valid && bar_unlock_sel) begin
            stalled_n = 0;
        end

        // an exiting warp is retired as its CTA accounting takes it, before
        // the slot can be relaunched
        if (retire_hit) begin
            active_n  = 0;
            tmask_n   = '0;
            stalled_n = 0;
        end

        // wsync unlock: warp pipeline drained
        if (wctl.wsync_valid && ctl_hit) begin
            stalled_n = 0;
        end

        if (br_hit) begin
            if (branch.is_trap) begin
                pc_n = from_fullPC(mtvec);
            end else if (branch.is_mret) begin
                // MRET/SRET/URET: restore the saved PC from mepc.
                pc_n = from_fullPC(mepc);
            end else if (branch.taken) begin
                pc_n = branch.dest;
            end
            stalled_n = 0;
        end

        // stall the warp until decode stage
        if (sched_fire && sched_sel) begin
            stalled_n = 1;
        `ifndef VX_CFG_EXT_C_ENABLE
            pc_n = pc_r + from_fullPC(`VX_CFG_XLEN'(4));
        `endif
        end

    `ifdef VX_CFG_EXT_C_ENABLE
        // Every redirect source stalls its warp from schedule until it resolves, so
        // no redirect writes a warp's PC on the cycle that warp decode-advances.
        if (dec_hit) begin
            pc_n = pc_r + from_fullPC(dec_is_rvc ? `VX_CFG_XLEN'(2) : `VX_CFG_XLEN'(4));
        end
    `endif

    `ifdef VX_CFG_EXT_RTU_ENABLE
        // A wstall'd TRACE retires: resume the warp so it proceeds to WAIT.
        if (rtu_unlock_hit) begin
            stalled_n = 1'b0;
        end
    `endif
    end

    always @(posedge clk) begin
        if (reset) begin
            active_r       <= 0;
            stalled_r      <= 0;
            tmask_r        <= '0;
            pc_r           <= '0;
            mscratch_r     <= '0;
            mstatus_mie_r  <= 1'b0;
            mstatus_mpie_r <= 1'b0;
            mstatus_fs_r   <= FS_RESET;
            mepc_r         <= '0;
            mcause_r       <= '0;
        end else begin
            active_r  <= active_n;
            stalled_r <= stalled_n;
            tmask_r   <= tmask_n;
            pc_r      <= pc_n;

            // spawned warps inherit the spawning warp's mscratch
            if (wspawn_apply && wspawn_sel && !wspawn_src_hit) begin
                mscratch_r <= wspawn_mscratch;
            end
            // CTA dispatch: latch this warp's mscratch (param)
            if (cta_hit) begin
                mscratch_r <= cta_param;
            end
            if (csr_wr.valid && csr_wr_hit) begin
                mscratch_r <= csr_wr.data;
            end

            if (dec_hit && dec_fs_dirty) begin
                mstatus_fs_r <= 2'b11;
            end

            if (csr_wr_hit) begin
                if (csr_wr.trap_valid[TRAP_CSR_MSTATUS]) begin
                    mstatus_mie_r  <= csr_wr.trap_data[`VX_CSR_MSTATUS_MIE];
                    mstatus_mpie_r <= csr_wr.trap_data[`VX_CSR_MSTATUS_MPIE];
                    mstatus_fs_r   <= csr_wr.trap_data[`VX_CSR_MSTATUS_FS +: 2] & FS_WMASK;
                end
                if (csr_wr.trap_valid[TRAP_CSR_MEPC]) begin
                    mepc_r <= csr_wr.trap_data[`VX_CFG_XLEN-1:MEPC_LSB];
                end
                if (csr_wr.trap_valid[TRAP_CSR_MCAUSE]) begin
                    mcause_r <= TRAP_CAUSE_BITS'(csr_wr.trap_data);
                end
            end

            // Hardware trap entry and return win a same-cycle software write.
            if (br_hit && branch.is_trap) begin
                mstatus_mie_r  <= 1'b0;
                mstatus_mpie_r <= mstatus_mie_r;
                mepc_r   <= (`VX_CFG_XLEN-MEPC_LSB)'(to_fullPC(branch.dest) >> MEPC_LSB);
                mcause_r <= branch.trap_cause;
            end
            if (br_hit && branch.is_mret) begin
                mstatus_mie_r  <= mstatus_mpie_r;
                mstatus_mpie_r <= 1'b1;
            end
        end
    end

    assign active   = active_r;
    assign stalled  = stalled_r;
    assign tmask    = tmask_r;
    assign pc       = pc_r;
    assign mscratch = mscratch_r;
    assign mcause   = mcause_r;

    // ibuffer occupancy: registered full keeps arbitration off the critical path
    logic [IBUF_CW-1:0] ibuf_size_r, ibuf_size_n;
    logic ibuf_full_r;
    wire ibuf_incr = sched_fire && sched_sel;
    assign ibuf_size_n = ibuf_size_r + IBUF_CW'(ibuf_incr) - IBUF_CW'(ibuf_pop);
    wire ibuf_full_n = (ibuf_size_n == IBUF_CW'(`VX_CFG_IBUF_SIZE));
    always @(posedge clk) begin
        if (reset) begin
            ibuf_size_r <= '0;
            ibuf_full_r <= 1'b0;
        end else begin
            ibuf_size_r <= ibuf_size_n;
            ibuf_full_r <= ibuf_full_n;
        end
    end
    assign ibuf_full = ibuf_full_r;

    VX_pending_size #(
        .SIZE      (256),
        .ALM_EMPTY (1)
    ) pending_ctr (
        .clk       (clk),
        .reset     (reset),
        .incr      (issued),
        .decr      (committed),
        .empty     (pending_empty),
        .alm_empty (pending_alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (active_n != active_r
         || (active_r && (stalled_n != stalled_r || tmask_n != tmask_r))) begin
            `TRACE(1, ("%t: %s warp-state: wid=%0d, active=%b, stalled=%b, tmask=%b\n",
                $time, INSTANCE_ID, wfs_to_wid(lid, slice_id), active_n, stalled_n, tmask_n))
        end
    end
`endif

endmodule
