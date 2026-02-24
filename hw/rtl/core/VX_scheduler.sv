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

    // configuration
    input base_dcrs_t       base_dcrs,

    // inputsdecode_if
    VX_warp_ctl_if.slave    warp_ctl_if,
`ifdef EXT_DXA_ENABLE
    VX_txbar_bus_if.slave   txbar_if,
`endif
    VX_branch_ctl_if.slave  branch_ctl_if [`NUM_ALU_BLOCKS],
    VX_decode_sched_if.slave decode_sched_if,
    VX_issue_sched_if.slave issue_sched_if,
    VX_commit_sched_if.slave commit_sched_if,

    // outputs
    VX_schedule_if.master   schedule_if,
`ifdef GBAR_ENABLE
    VX_gbar_bus_if.master   gbar_bus_if,
`endif
    VX_sched_csr_if.master  sched_csr_if,

    // status
    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    reg [`NUM_WARPS-1:0] active_warps, active_warps_n; // updated when a warp is activated or disabled
    reg [`NUM_WARPS-1:0] stalled_warps, stalled_warps_n;  // set when branch/gpgpu instructions are issued

    reg [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks, thread_masks_n;
    reg [`NUM_WARPS-1:0][PC_BITS-1:0] warp_pcs, warp_pcs_n;

    wire [NW_WIDTH-1:0]     schedule_wid;
    wire [`NUM_THREADS-1:0] schedule_tmask;
    wire [PC_BITS-1:0]      schedule_pc;
    wire                    schedule_valid;
    wire                    schedule_ready;

    // split/join
    wire                    join_valid;
    wire                    join_is_dvg;
    wire                    join_is_else;
    wire [NW_WIDTH-1:0]     join_wid;
    wire [`NUM_THREADS-1:0] join_tmask;
    wire [PC_BITS-1:0]      join_pc;

    reg [PERF_CTR_BITS-1:0] cycles;

    wire schedule_fire = schedule_valid && schedule_ready;
    wire schedule_if_fire = schedule_if.valid && schedule_if.ready;

    // branch
    wire [`NUM_ALU_BLOCKS-1:0]               branch_valid;
    wire [`NUM_ALU_BLOCKS-1:0][NW_WIDTH-1:0] branch_wid;
    wire [`NUM_ALU_BLOCKS-1:0]               branch_taken;
    wire [`NUM_ALU_BLOCKS-1:0][PC_BITS-1:0]  branch_dest;
    for (genvar i = 0; i < `NUM_ALU_BLOCKS; ++i) begin : g_branch_init
        assign branch_valid[i] = branch_ctl_if[i].valid;
        assign branch_wid[i]   = branch_ctl_if[i].wid;
        assign branch_taken[i] = branch_ctl_if[i].taken;
        assign branch_dest[i]  = branch_ctl_if[i].dest;
    end

    // barriers
    wire [`NUM_WARPS-1:0] barrier_unlock_mask;
    wire barrier_unlock_valid;

    // wspawn
    wspawn_t wspawn;
    reg [NW_WIDTH-1:0] wspawn_wid;
    reg is_single_warp;

    wire [`CLOG2(`NUM_WARPS+1)-1:0] active_warps_cnt;
    `POP_COUNT(active_warps_cnt, active_warps);

     always @(*) begin
        active_warps_n  = active_warps;
        stalled_warps_n = stalled_warps;
        thread_masks_n  = thread_masks;
        warp_pcs_n      = warp_pcs;

        // decode unlock
        if (decode_sched_if.valid && decode_sched_if.unlock) begin
            stalled_warps_n[decode_sched_if.wid] = 0;
        end

        // wspawn handling
        if (wspawn.valid && is_single_warp) begin
            active_warps_n |= wspawn.wmask;
            for (integer i = 0; i < `NUM_WARPS; ++i) begin
                if (wspawn.wmask[i]) begin
                    thread_masks_n[i][0] = 1;
                    warp_pcs_n[i] = wspawn.pc;
                end
            end
            stalled_warps_n[wspawn_wid] = 0; // unlock warp
        end

        // TMC handling
        if (warp_ctl_if.valid && warp_ctl_if.tmc.valid) begin
            active_warps_n[warp_ctl_if.wid]  = (warp_ctl_if.tmc.tmask != 0);
            thread_masks_n[warp_ctl_if.wid]  = warp_ctl_if.tmc.tmask;
            stalled_warps_n[warp_ctl_if.wid] = 0; // unlock warp
        end

        // split handling
        if (warp_ctl_if.valid && warp_ctl_if.split.valid) begin
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
        if (barrier_unlock_valid) begin
            stalled_warps_n &= ~barrier_unlock_mask;
        end

        // Branch handling
        for (integer i = 0; i < `NUM_ALU_BLOCKS; ++i) begin
            if (branch_valid[i]) begin
                if (branch_taken[i]) begin
                    warp_pcs_n[branch_wid[i]] = branch_dest[i];
                end
                stalled_warps_n[branch_wid[i]] = 0; // unlock warp
            end
        end

        // stall the warp until decode stage
        if (schedule_fire) begin
            stalled_warps_n[schedule_wid] = 1;
        end

        // advance PC
        if (schedule_if_fire) begin
            warp_pcs_n[schedule_if.data.wid] = schedule_if.data.PC + from_fullPC(`XLEN'(4));
        end
    end

    `UNUSED_VAR (base_dcrs)

    always @(posedge clk) begin
        if (reset) begin
            stalled_warps   <= '0;
            warp_pcs        <= '0;
            active_warps    <= '0;
            thread_masks    <= '0;
            cycles          <= '0;
            wspawn.valid    <=  0;
        `ifdef RVTEST_MT
            // activate all warps as single-thread
            for (integer i = 0; i < `NUM_WARPS; ++i) begin
                warp_pcs[i]     <= from_fullPC(base_dcrs.startup_addr);
                active_warps[i] <= 1;
                thread_masks[i][0] <= 1;
            end
        `else
            // activate first warp as single-thread
            warp_pcs[0]     <= from_fullPC(base_dcrs.startup_addr);
            active_warps[0] <= 1;
            thread_masks[0][0] <= 1;
        `endif
            is_single_warp  <= 1;
        end else begin
            active_warps   <= active_warps_n;
            stalled_warps  <= stalled_warps_n;
            thread_masks   <= thread_masks_n;
            warp_pcs       <= warp_pcs_n;
            is_single_warp <= (active_warps_cnt == $bits(active_warps_cnt)'(1));

            // wspawn handling
            if (warp_ctl_if.valid && warp_ctl_if.wspawn.valid) begin
                wspawn.valid <= 1;
                wspawn.wmask <= warp_ctl_if.wspawn.wmask;
                wspawn.pc    <= warp_ctl_if.wspawn.pc;
                wspawn_wid   <= warp_ctl_if.wid;
            end
            if (wspawn.valid && is_single_warp) begin
                wspawn.valid <= 0;
            end

            if (busy) begin
                cycles <= cycles + 1;
            end
        end
    end

    // barrier handling
`ifdef EXT_DXA_ENABLE
    wire bar_req_data_valid = warp_ctl_if.valid && warp_ctl_if.barrier.valid;
    wire txbar_fire = txbar_if.valid && txbar_if.ready;
`endif

    VX_bar_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-barrier", INSTANCE_ID))),
        .CORE_ID     (CORE_ID)
    ) bar_unit (
        .clk        (clk),
        .reset      (reset),
        .req_valid  (warp_ctl_if.valid),
        .req_wid    (warp_ctl_if.wid),
        .req_data   (warp_ctl_if.barrier),
    `ifdef EXT_DXA_ENABLE
        .tx_valid   (txbar_fire),
        .tx_bar_addr(txbar_if.data.addr),
        .tx_is_done (txbar_if.data.is_done),
    `else
        .tx_valid   (1'b0),
        .tx_bar_addr('0),
        .tx_is_done (1'b0),
    `endif
        .read_addr  (warp_ctl_if.barrier_addr),
        .read_phase (warp_ctl_if.barrier_phase),
        .active_warps(active_warps),
    `ifdef GBAR_ENABLE
        .gbar_bus_if(gbar_bus_if),
    `endif
        .unlock_valid(barrier_unlock_valid),
        .unlock_mask(barrier_unlock_mask)
    );

`ifdef EXT_DXA_ENABLE
    assign txbar_if.ready = ~bar_req_data_valid;
`endif

    // split/join handling

    VX_split_join #(
        .INSTANCE_ID (`SFORMATF(("%s-splitjoin", INSTANCE_ID))),
        .OUT_REG     (1)
    ) split_join (
        .clk        (clk),
        .reset      (reset),
        .valid      (warp_ctl_if.valid),
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

    wire [`NUM_WARPS-1:0] ready_warps = active_warps & ~stalled_warps;

    VX_priority_encoder #(
        .N (`NUM_WARPS)
    ) wid_select (
        .data_in   (ready_warps),
        .index_out (schedule_wid),
        .valid_out (schedule_valid),
        `UNUSED_PIN (onehot_out)
    );

    wire [`NUM_WARPS-1:0][(`NUM_THREADS + PC_BITS)-1:0] schedule_data;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin : g_schedule_data
        assign schedule_data[i] = {thread_masks[i], warp_pcs[i]};
    end

    assign {schedule_tmask, schedule_pc} = {
        schedule_data[schedule_wid][(`NUM_THREADS + PC_BITS)-1:(`NUM_THREADS + PC_BITS)-4],
        schedule_data[schedule_wid][(`NUM_THREADS + PC_BITS)-5:0]
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

    VX_elastic_buffer #(
        .DATAW (`NUM_THREADS + PC_BITS + NW_WIDTH + UUID_WIDTH),
        .SIZE  (2),  // need to buffer out ready_in
        .OUT_REG (1) // should be registered for BRAM acces in fetch unit
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (schedule_valid),
        .ready_in  (schedule_ready),
        .data_in   ({schedule_tmask, schedule_pc, schedule_wid, instr_uuid}),
        .data_out  ({schedule_if.data.tmask, schedule_if.data.PC, schedule_if.data.wid, schedule_if.data.uuid}),
        .valid_out (schedule_if.valid),
        .ready_out (schedule_if.ready)
    );

    // Track committed instructions

    reg [PERF_CTR_BITS-1:0] instret;

    always @(posedge clk) begin
        if (reset) begin
            instret <= '0;
        end else begin
            instret <= instret + PERF_CTR_BITS'(commit_sched_if.committed_warps_cnt);
        end
    end

    // Track pending instructions

    wire pending_warp_empty;

    VX_pending_size #(
        .SIZE      (2048),
        .INCRW     (ISSUE_ISW_SIZEW),
        .DECRW     (ISSUE_ISW_SIZEW)
    ) counter (
        .clk       (clk),
        .reset     (reset),
        .incr      (issue_sched_if.issued_warps_cnt),
        .decr      (commit_sched_if.committed_warps_cnt),
        .empty     (pending_warp_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    `BUFFER_EX(busy, (active_warps != 0 || ~pending_warp_empty), 1'b1, 1, 1);

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

    `RUNTIME_ASSERT(timeout_ctr < STALL_TIMEOUT, ("*** %s timeout: active_warps=%b, stalled_warps=%b", INSTANCE_ID, active_warps, stalled_warps))

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_sched_idles;
    reg [PERF_CTR_BITS-1:0] perf_active_warps;
    reg [PERF_CTR_BITS-1:0] perf_stalled_warps;
    reg [PERF_CTR_BITS-1:0] perf_issued_warps;
    reg [PERF_CTR_BITS-1:0] perf_issued_threads;
    reg [PERF_CTR_BITS-1:0] perf_branches;
    reg [PERF_CTR_BITS-1:0] perf_divergence;

    wire [`CLOG2(`NUM_WARPS+1)-1:0] stalled_warps_cnt;
    wire [`CLOG2(`NUM_ALU_BLOCKS+1)-1:0] branches_cnt;
    wire [`CLOG2(`NUM_THREADS+1)-1:0] issued_threads_cnt;

    wire schedule_idle = ~schedule_valid;
    wire has_divergence = warp_ctl_if.valid && warp_ctl_if.split.valid && warp_ctl_if.split.is_dvg;
    wire [`NUM_THREADS-1:0] issued_threads = {`NUM_THREADS{schedule_if_fire}} & schedule_if.data.tmask;

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
    for (genvar w = 0; w < `NUM_WARPS; ++w) begin : g_trace_warp_status
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
            `TRACE(1, ("%t: %s dispatch: wid=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_wid, to_fullPC(schedule_pc), schedule_tmask, instr_uuid))
        end
    end
`endif

endmodule
