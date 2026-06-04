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
    reg [`VX_CFG_NUM_WARPS-1:0] stalled_warps, stalled_warps_n;  // set when branch/gpgpu instructions are issued

    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] thread_masks, thread_masks_n;
    reg [`VX_CFG_NUM_WARPS-1:0][PC_BITS-1:0] warp_pcs, warp_pcs_n;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch_r;
    reg [`VX_CFG_NUM_WARPS-1:0][NCTA_WIDTH-1:0] cta_id_per_warp_r;

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
    cta_csrs_t cta_csrs;
    wire cta_dispatcher_busy;
    wire cta_init;

    // Per-CTA context.
    wire                  cta_ctx_write;
    wire [NCTA_WIDTH-1:0] cta_ctx_waddr;
    cta_ctx_t             cta_ctx_wdata;
    wire [NCTA_WIDTH-1:0] cta_ctx_raddr;
    cta_ctx_t             cta_ctx_rdata;

    VX_dp_ram #(
        .DATAW     ($bits(cta_ctx_t)),
        .SIZE      (NUM_CTA_MAX),
        .RDW_MODE  ("R"),
        .RADDR_REG (1)
    ) cta_ctx_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cta_ctx_write),
        .wren  (1'b1),
        .waddr (cta_ctx_waddr),
        .wdata (cta_ctx_wdata),
        .raddr (cta_ctx_raddr),
        .rdata (cta_ctx_rdata)
    );

    // Per-warp residue.
    wire                cta_warp_write;
    wire [NW_WIDTH-1:0] cta_warp_waddr;
    cta_warp_t          cta_warp_wdata;
    wire [NW_WIDTH-1:0] cta_warp_raddr;
    cta_warp_t          cta_warp_rdata;

    VX_dp_ram #(
        .DATAW     ($bits(cta_warp_t)),
        .SIZE      (`VX_CFG_NUM_WARPS),
        .RDW_MODE  ("R"),
        .RADDR_REG (1)
    ) cta_warp_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cta_warp_write),
        .wren  (1'b1),
        .waddr (cta_warp_waddr),
        .wdata (cta_warp_wdata),
        .raddr (cta_warp_raddr),
        .rdata (cta_warp_rdata)
    );

    // Warp retirement: TMC with tmask==0 permanently deactivates the warp
    wire cta_warp_done = warp_ctl_if.tmc_valid && (warp_ctl_if.tmc.tmask == 0);

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
        .cta_csrs   (cta_csrs),
        .cta_init   (cta_init),
        .busy       (cta_dispatcher_busy)
    );

    assign sched_csr_if.mscratch  = mscratch_r[sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mstatus = mstatus_r[sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mtvec   = mtvec_r  [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mepc    = mepc_r   [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mcause  = mcause_r [sched_csr_if.csr_rd_wid];
    assign sched_csr_if.csr_mtval   = mtval_r  [sched_csr_if.csr_rd_wid];

    // -----------------------------------------------------------------------
    // Per-CTA + per-warp sp_ram drivers
    // -----------------------------------------------------------------------

    assign cta_warp_write       = cta_fire;
    assign cta_warp_waddr       = cta_wid;
    assign cta_warp_wdata.cta_rank   = cta_csrs.cta_rank;
    assign cta_warp_wdata.thread_idx = cta_csrs.thread_idx;

    assign cta_ctx_write        = cta_fire;
    assign cta_ctx_waddr        = cta_csrs.cta_id;
    assign cta_ctx_wdata.cta_size  = cta_csrs.cta_size;
    assign cta_ctx_wdata.block_idx = cta_csrs.block_idx;
    assign cta_ctx_wdata.block_dim = cta_csrs.block_dim;
    assign cta_ctx_wdata.grid_dim  = cta_csrs.grid_dim;
    assign cta_ctx_wdata.param     = cta_csrs.param;
    assign cta_ctx_wdata.lmem_addr = cta_csrs.lmem_addr;
    assign cta_ctx_wdata.cluster_size = cta_csrs.cluster_size;
    assign cta_ctx_wdata.entry     = cta_csrs.entry;

    // sp_ram returns rdata one cycle after raddr; csr_unit holds execute_if stable
    // for one cycle so outputs are valid when consumed.
    assign cta_warp_raddr = sched_csr_if.csr_rd_wid;
    assign cta_ctx_raddr  = sched_csr_if.csr_rd_cta_id;

    assign sched_csr_if.cta_csrs.cta_id     = sched_csr_if.csr_rd_cta_id;
    assign sched_csr_if.cta_csrs.cta_rank   = cta_warp_rdata.cta_rank;
    assign sched_csr_if.cta_csrs.thread_idx = cta_warp_rdata.thread_idx;
    assign sched_csr_if.cta_csrs.cta_size   = cta_ctx_rdata.cta_size;
    assign sched_csr_if.cta_csrs.block_idx  = cta_ctx_rdata.block_idx;
    assign sched_csr_if.cta_csrs.block_dim  = cta_ctx_rdata.block_dim;
    assign sched_csr_if.cta_csrs.grid_dim   = cta_ctx_rdata.grid_dim;
    assign sched_csr_if.cta_csrs.param      = cta_ctx_rdata.param;
    assign sched_csr_if.cta_csrs.lmem_addr  = cta_ctx_rdata.lmem_addr;
    assign sched_csr_if.cta_csrs.cluster_size = cta_ctx_rdata.cluster_size;
    assign sched_csr_if.cta_csrs.entry      = cta_ctx_rdata.entry;

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

        // dispatch warps
        if (cta_fire) begin
            active_warps_n[cta_wid] = 1;
            // Reusing a warp for the next CTA skips the one-time prologue and rewinds to the
            // kernel's per-CTA dispatch window: a fixed 20-byte (5-instruction) sequence that
            // reloads the entry pointer and kargs before re-calling.
            warp_pcs_n[cta_wid] = cta_init ? cta_PC : (warp_pcs[cta_wid] - from_fullPC(`VX_CFG_XLEN'(20)));
            thread_masks_n[cta_wid] = cta_tmask;
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

        // TMC handling
        if (warp_ctl_if.tmc_valid) begin
            active_warps_n[warp_ctl_if.wid]  = (warp_ctl_if.tmc.tmask != 0);
            thread_masks_n[warp_ctl_if.wid]  = warp_ctl_if.tmc.tmask;
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
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

        // Branch handling
        for (integer i = 0; i < `VX_CFG_NUM_ALU_BLOCKS; ++i) begin
            if (branch_valid[i]) begin
                if (branch_is_trap[i]) begin
                    // ECALL/EBREAK: redirect to trap vector (mtvec[1:0] = MODE field; mask off to get base address).
                    warp_pcs_n[branch_wid[i]] = from_fullPC(mtvec_r[branch_wid[i]] & ~`VX_CFG_XLEN'(3));
                end else if (branch_is_mret[i]) begin
                    // MRET/SRET/URET: restore the saved PC from mepc.
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
        // With RVC, the decompressor may emit a 2-byte instruction.
        if (decode_sched_if.valid) begin
            warp_pcs_n[decode_sched_if.wid] =
                warp_pcs_n[decode_sched_if.wid]
                + from_fullPC(decode_sched_if.is_rvc ? `VX_CFG_XLEN'(2) : `VX_CFG_XLEN'(4));
        end
    `else
        if (schedule_if_fire) begin
            warp_pcs_n[schedule_if.data.wid] = schedule_if.data.PC + from_fullPC(`VX_CFG_XLEN'(4));
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
        end else begin
            active_warps   <= active_warps_n;
            stalled_warps  <= stalled_warps_n;
            thread_masks   <= thread_masks_n;
            warp_pcs       <= warp_pcs_n;
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

            // CTA dispatch: latch this warp's cta_id and mscratch (param) in
            // flops; the per-CTA and per-warp tables (cta_ctx_ram /
            // cta_warp_ram) are written via their dedicated write ports below.
            if (cta_fire) begin
                mscratch_r[cta_wid] <= cta_csrs.param;
                cta_id_per_warp_r[cta_wid] <= cta_csrs.cta_id;
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

    // Look up the scheduled warp's CTA-id from the per-warp table.
    wire [NCTA_WIDTH-1:0] schedule_cta_id = cta_id_per_warp_r[schedule_wid];

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
