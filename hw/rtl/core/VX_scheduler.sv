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

// The core-level part of scheduling: CTA dispatch and the core-wide warp-control
// resources. Each frontend slice schedules and fetches its own warps (VX_warp_sched,
// VX_fetch); core-wide events are handed to it with slice-local warp ids.
module VX_scheduler import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

`ifdef PERF_ENABLE
    output sched_perf_t     sched_perf,
`endif

    VX_slice_sched_if.master slice_if [`VX_CFG_FETCH_WIDTH],
    VX_warp_shared_if.slave warp_shared_if [`VX_CFG_NUM_SFU_BLOCKS],
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_txbar_bus_if.slave   txbar_bus_if,
`endif
    input wire [`VX_CFG_NUM_WARPS-1:0] committed_warps,

    // CTA context writes, to every SFU block's copy
    output cta_bcast_t      cta_bcast,

    VX_kmu_bus_if.slave     kmu_bus_if,

    VX_gbar_bus_if.master   gbar_bus_if,

    // per-core CSR state
    output wire [PERF_CTR_BITS-1:0] cycles,
    output wire [PERF_CTR_BITS-1:0] instret,
    output wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
`ifdef VX_CFG_VM_ENABLE
    output wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] satp,
    input wire [`VX_CFG_NUM_SFU_BLOCKS-1:0] satp_wr_valid,
    input wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] satp_wr_data,
`endif

    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam FW  = `VX_CFG_FETCH_WIDTH;
    localparam SW  = PER_FETCH_WARPS;

    // slice status in wid order (wid = {wfs, fsw})
    wire [`VX_CFG_NUM_WARPS-1:0] stalled_warps;
    wire [`VX_CFG_NUM_WARPS-1:0] ibuf_full_n, pending_warp_empty;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_interleave
        assign active_warps[w]       = slice_if[w % FW].active[w / FW];
        assign stalled_warps[w]      = slice_if[w % FW].stalled[w / FW];
        assign ibuf_full_n[w]        = slice_if[w % FW].ibuf_full_n[w / FW];
        assign pending_warp_empty[w] = slice_if[w % FW].pending_empty[w / FW];
    end

    // CTA dispatcher: lane p launches frontend slice p's warps
    wire [FW-1:0] cta_fire;
    wire [FW-1:0][NW_WIDTH-1:0] cta_wid;
    wire [PC_BITS-1:0] cta_PC;
    wire [FW-1:0][`VX_CFG_NUM_THREADS-1:0] cta_tmask;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param;
    wire [FW-1:0] cta_init;
    cta_lane_bcast_t [FW-1:0] cta_lane_bcast;
    wire cta_dispatcher_busy;
    wire cta_warp_done;
    wire [NW_WIDTH-1:0] cta_warp_done_wid;

    VX_cta_dispatch #(
        .INSTANCE_ID (`SFORMATF(("%s-cta_dispatch", INSTANCE_ID))),
        .NUM_LANES   (FW)
    ) cta_dispatcher (
        .clk          (clk),
        .reset        (reset),
        .kmu_bus_if   (kmu_bus_if),
        .active_warps (active_warps),
        .warp_done    (cta_warp_done),
        .warp_done_wid(cta_warp_done_wid),
        .cta_fire     (cta_fire),
        .cta_wid      (cta_wid),
        .cta_PC       (cta_PC),
        .cta_tmask    (cta_tmask),
        .cta_param    (cta_param),
        .cta_init     (cta_init),
        .cta_bcast    (cta_bcast),
        .cta_lane_bcast(cta_lane_bcast),
        .busy         (cta_dispatcher_busy)
    );

    // core-wide warp-control resources
    wire bar_unlock_valid;
    wire [`VX_CFG_NUM_WARPS-1:0] bar_unlock_mask;
    wire wspawn_apply;
    wire [`VX_CFG_NUM_WARPS-1:0] wspawn_wmask;
    wire [PC_BITS-1:0] wspawn_pc;
    wire [NW_WIDTH-1:0] wspawn_wid;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] wspawn_mscratch;

    VX_warp_shared_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_REQS    (`VX_CFG_NUM_SFU_BLOCKS)
    ) warp_shared_unit (
        .clk             (clk),
        .reset           (reset),
        .core_id         (core_id),
        .warp_shared_if  (warp_shared_if),
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .txbar_bus_if    (txbar_bus_if),
    `endif
        .active_warps    (active_warps),
        .bar_unlock_valid(bar_unlock_valid),
        .bar_unlock_mask (bar_unlock_mask),
        .gbar_bus_if     (gbar_bus_if),
        .wspawn_apply    (wspawn_apply),
        .wspawn_wmask    (wspawn_wmask),
        .wspawn_pc       (wspawn_pc),
        .wspawn_wid      (wspawn_wid),
        .wspawn_mscratch (wspawn_mscratch),
        .warp_done       (cta_warp_done),
        .warp_done_wid   (cta_warp_done_wid),
        `UNUSED_PIN      (wspawn_pending)
    );

    // counters

    reg [PERF_CTR_BITS-1:0] cycles_r;
    reg [PERF_CTR_BITS-1:0] instret_r;
    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] committed_warps_cnt;
    `POP_COUNT(committed_warps_cnt, committed_warps);
    always @(posedge clk) begin
        if (reset) begin
            cycles_r  <= '0;
            instret_r <= '0;
        end else begin
            if (busy) begin
                cycles_r <= cycles_r + 1;
            end
            instret_r <= instret_r + PERF_CTR_BITS'(committed_warps_cnt);
        end
    end
    assign cycles  = cycles_r;
    assign instret = instret_r;

    // from the slices' next-state full, so it is valid the same cycle as their full flags
    reg all_ibuf_full;
    always @(posedge clk) begin
        if (reset) begin
            all_ibuf_full <= 1'b0;
        end else begin
            all_ibuf_full <= (& ibuf_full_n);
        end
    end

`ifdef VX_CFG_VM_ENABLE
    // Per-core SATP CSR, written by any SFU block's CSR unit. Accesses in one
    // cycle take effect in block order: each block reads the value the lower
    // blocks leave.
    reg [`VX_CFG_XLEN-1:0] satp_r;
    wire [`VX_CFG_NUM_SFU_BLOCKS:0][`VX_CFG_XLEN-1:0] satp_chain;
    assign satp_chain[0] = satp_r;
    for (genvar b = 0; b < `VX_CFG_NUM_SFU_BLOCKS; ++b) begin : g_satp
        assign satp[b] = satp_chain[b];
        assign satp_chain[b+1] = satp_wr_valid[b] ? satp_wr_data[b] : satp_chain[b];
    end
    always @(posedge clk) begin
        if (reset) begin
            satp_r <= '0;
        end else begin
            satp_r <= satp_chain[`VX_CFG_NUM_SFU_BLOCKS];
        end
    end
`endif

    // per-slice view of the core-wide events

    wire [FETCH_FSW_W-1:0] wspawn_fsw = wid_to_fsw(wspawn_wid);

    for (genvar p = 0; p < FW; ++p) begin : g_slices
        localparam [FETCH_FSW_W-1:0] FSW = FETCH_FSW_W'(p);

        assign slice_if[p].cta_fire     = cta_fire[p];
        assign slice_if[p].cta_lid      = wid_to_wfs(cta_wid[p]);
        assign slice_if[p].cta_PC       = cta_PC;
        assign slice_if[p].cta_tmask    = cta_tmask[p];
        assign slice_if[p].cta_param    = cta_param;
        assign slice_if[p].cta_init     = cta_init[p];

        assign slice_if[p].warp_write   = cta_lane_bcast[p].warp_write;
        assign slice_if[p].warp_waddr   = wid_to_wfs(cta_lane_bcast[p].warp_waddr);
        assign slice_if[p].warp_wdata   = cta_lane_bcast[p].warp_wdata;
        assign slice_if[p].id_write     = cta_lane_bcast[p].id_write;
        assign slice_if[p].id_waddr     = wid_to_wfs(cta_lane_bcast[p].id_wid);
        assign slice_if[p].id_wdata     = cta_lane_bcast[p].id_cta;
        `UNUSED_VAR ({wid_to_fsw(cta_wid[p]), wid_to_fsw(cta_lane_bcast[p].warp_waddr), wid_to_fsw(cta_lane_bcast[p].id_wid)})

        assign slice_if[p].wspawn_apply    = wspawn_apply;
        assign slice_if[p].wspawn_pc       = wspawn_pc;
        assign slice_if[p].wspawn_src      = (wspawn_fsw == FSW);
        assign slice_if[p].wspawn_src_lid  = wid_to_wfs(wspawn_wid);
        assign slice_if[p].wspawn_mscratch = wspawn_mscratch;

        assign slice_if[p].bar_unlock_valid = bar_unlock_valid;

        assign slice_if[p].retire_valid = cta_warp_done && (wid_to_fsw(cta_warp_done_wid) == FSW);
        assign slice_if[p].retire_lid   = wid_to_wfs(cta_warp_done_wid);

        for (genvar l = 0; l < SW; ++l) begin : g_lw
            assign slice_if[p].wspawn_wmask[l]    = wspawn_wmask[l * FW + p];
            assign slice_if[p].bar_unlock_mask[l] = bar_unlock_mask[l * FW + p];
        end

        assign slice_if[p].all_ibuf_full = all_ibuf_full;
    end

    wire [FW-1:0] sched_valid, sched_fire;
    wire [FW-1:0][`VX_CFG_NUM_THREADS-1:0] sched_tmask;
    for (genvar p = 0; p < FW; ++p) begin : g_sched_status
        assign sched_valid[p] = slice_if[p].sched_valid;
        assign sched_fire[p]  = slice_if[p].sched_fire;
        assign sched_tmask[p] = slice_if[p].sched_tmask;
    end
`ifndef PERF_ENABLE
    `UNUSED_VAR ({sched_valid, sched_fire, sched_tmask})
`endif

    // active_warps is the slices' registered state, so a CTA launch shows there one cycle after
    // cta_fire; counting the launch itself keeps busy from dropping between the dispatcher going
    // idle and the launched warps becoming visible (a CTA launched in a single cycle).
    wire busy_buf;
    `BUFFER_EX(busy_buf, (active_warps != 0 || (| cta_fire) || ~(&pending_warp_empty)), 1'b1, 1, 1);
    assign busy = busy_buf || cta_dispatcher_busy;

    // timeout handling
    reg [31:0] timeout_ctr;
    reg timeout_enable;
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr    <= '0;
            timeout_enable <= 0;
        end else begin
            if (| cta_fire) begin
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
    wire timeout_dump = !reset && (timeout_ctr == (SCHED_STALL_TIMEOUT - 1));
    always @(posedge clk) begin
        if (timeout_dump) begin
            $display("*** %s scheduler-timeout dump: active=%b stalled=%b", INSTANCE_ID, active_warps, stalled_warps);
        end
    end
    for (genvar p = 0; p < FW; ++p) begin : g_timeout_dump
        assign slice_if[p].timeout_dump = timeout_dump;
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

    localparam NPA = FETCH_ALU_PORTS;
    localparam NPS = FETCH_SFU_PORTS;

    wire [FW-1:0][NPA-1:0] slice_branch;
    wire [FW-1:0][NPS-1:0] slice_divergence;
    for (genvar p = 0; p < FW; ++p) begin : g_perf_slices
        assign slice_branch[p]     = slice_if[p].branch_valid;
        assign slice_divergence[p] = slice_if[p].divergence;
    end

    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] active_warps_cnt, stalled_warps_cnt;
    wire [`CLOG2(FW*NPA+1)-1:0] branches_cnt;
    wire [`CLOG2(FW*NPS+1)-1:0] divergence_cnt;
    wire [`CLOG2(FW+1)-1:0] issued_warps_cnt;
    wire [`CLOG2(FW*`VX_CFG_NUM_THREADS+1)-1:0] issued_threads_cnt;
    wire [FW-1:0][`VX_CFG_NUM_THREADS-1:0] issued_threads;
    for (genvar p = 0; p < FW; ++p) begin : g_issued_threads
        assign issued_threads[p] = {`VX_CFG_NUM_THREADS{sched_fire[p]}} & sched_tmask[p];
    end

    `POP_COUNT(active_warps_cnt, active_warps);
    `POP_COUNT(stalled_warps_cnt, stalled_warps);
    `POP_COUNT(issued_threads_cnt, issued_threads);
    `POP_COUNT(issued_warps_cnt, sched_fire);
    `POP_COUNT(branches_cnt, slice_branch);
    `POP_COUNT(divergence_cnt, slice_divergence);

    wire sched_idle = ~(| sched_valid);

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
            perf_sched_idles   <= perf_sched_idles + PERF_CTR_BITS'(sched_idle);
            perf_active_warps  <= perf_active_warps + PERF_CTR_BITS'(active_warps_cnt);
            perf_stalled_warps <= perf_stalled_warps + PERF_CTR_BITS'(stalled_warps_cnt);
            perf_issued_warps  <= perf_issued_warps + PERF_CTR_BITS'(issued_warps_cnt);
            perf_issued_threads<= perf_issued_threads + PERF_CTR_BITS'(issued_threads_cnt);
            perf_branches      <= perf_branches + PERF_CTR_BITS'(branches_cnt);
            perf_divergence    <= perf_divergence + PERF_CTR_BITS'(divergence_cnt);
        end
    end

    assign sched_perf.idles         = perf_sched_idles;
    assign sched_perf.active_warps  = perf_active_warps;
    assign sched_perf.stalled_warps = perf_stalled_warps;
    assign sched_perf.issued_warps  = perf_issued_warps;
    assign sched_perf.issued_threads= perf_issued_threads;
    assign sched_perf.branches      = perf_branches;
    assign sched_perf.divergence    = perf_divergence;
`else
    for (genvar p = 0; p < FW; ++p) begin : g_perf_unused
        `UNUSED_VAR ({slice_if[p].branch_valid, slice_if[p].divergence})
    end
`endif

endmodule
