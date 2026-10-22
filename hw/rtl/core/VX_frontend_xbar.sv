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

// Connects the frontend slices (frontend slice f owns wid mod FETCH_WIDTH == f) to the
// issue slices (wid mod ISSUE_WIDTH) and the ALU/SFU blocks (wid mod B). Issue
// slice i takes its instructions from frontend slice i mod FETCH_WIDTH. A block
// serving several frontend slices sends each its own warps' events and answers its
// reads from the slice owning the read wid. Purely combinational.
module VX_frontend_xbar import VX_gpu_pkg::*; (
    // instruction path
    VX_decode_if.slave      fs_decode_if [`VX_CFG_FETCH_WIDTH],
    VX_decode_if.master     is_decode_if [`VX_CFG_ISSUE_WIDTH],

    // issue and commit events, and per-warp drain status
    input wire [`VX_CFG_ISSUE_WIDTH-1:0] issue_valid,
    input wire [`VX_CFG_ISSUE_WIDTH-1:0][ISSUE_WIS_W-1:0] issue_wis,
    input wire [`VX_CFG_NUM_WARPS-1:0] committed_warps,
    output wire [`VX_CFG_FETCH_WIDTH-1:0][PER_FETCH_WARPS-1:0] fs_issued,
    output wire [`VX_CFG_FETCH_WIDTH-1:0][PER_FETCH_WARPS-1:0] fs_committed,
    input wire [`VX_CFG_FETCH_WIDTH-1:0][PER_FETCH_WARPS-1:0] fs_pending_alm_empty,

    // FU blocks
    VX_branch_ctl_if.slave  branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS],
    VX_warp_ctl_if.slave    warp_ctl_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_sched_csr_if.master  sched_csr_if [`VX_CFG_NUM_SFU_BLOCKS],
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_sched_unlock_if.slave sched_unlock_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

    // core scheduler state
    input wire [PERF_CTR_BITS-1:0] cycles,
    input wire [PERF_CTR_BITS-1:0] instret,
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    input wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_sched_empty,
    input wire              mmu_empty,
`ifdef VX_CFG_VM_ENABLE
    input wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] satp,
    output wire [`VX_CFG_NUM_SFU_BLOCKS-1:0] satp_wr_valid,
    output wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] satp_wr_data,
`endif

    // frontend slices
    output sched_branch_t [`VX_CFG_FETCH_WIDTH-1:0][FETCH_ALU_PORTS-1:0] fs_branch,
    output sched_wctl_t [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0] fs_wctl,
    output wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] fs_dvstack_wid,
    input wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0][DV_STACK_SIZEW-1:0] fs_dvstack_ptr,
    input wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] fs_dvstack_mscratch,
    output wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] fs_csr_rd_wid,
    input sched_csr_rd_t [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0] fs_csr_rd,
`ifdef VX_CFG_EXT_RTU_ENABLE
    output wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0] fs_rtu_unlock_valid,
    output wire [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0][NW_WIDTH-1:0] fs_rtu_unlock_wid,
`endif
    output sched_csr_wr_t [`VX_CFG_FETCH_WIDTH-1:0][FETCH_SFU_PORTS-1:0] fs_csr_wr
);
    localparam FW  = `VX_CFG_FETCH_WIDTH;
    localparam IW  = `VX_CFG_ISSUE_WIDTH;
    localparam NW  = `VX_CFG_NUM_WARPS;
    localparam SW  = PER_FETCH_WARPS;
    localparam BA  = `VX_CFG_NUM_ALU_BLOCKS;
    localparam BS  = `VX_CFG_NUM_SFU_BLOCKS;
    localparam NPA = FETCH_ALU_PORTS;
    localparam NPS = FETCH_SFU_PORTS;

    `STATIC_ASSERT (`IS_POW2(FW) && (FW <= IW) && ((IW % FW) == 0), ("invalid parameter"))

    // instruction path

    wire [IW-1:0] is_ready;
    for (genvar i = 0; i < IW; ++i) begin : g_issue_slices
        localparam F = i % FW;
        assign is_decode_if[i].valid = fs_decode_if[F].valid && (wid_to_isw(fs_decode_if[F].data.wid) == ISSUE_ISW_W'(i));
        assign is_decode_if[i].data  = fs_decode_if[F].data;
        assign is_ready[i]           = is_decode_if[i].ready;
        if (PER_ISSUE_WARPS < NW) begin : g_ibuf_pop_unused
            `UNUSED_VAR (is_decode_if[i].ibuf_pop[NW-1:PER_ISSUE_WARPS])
        end
    end

    for (genvar f = 0; f < FW; ++f) begin : g_frontend_slices
        assign fs_decode_if[f].ready = is_ready[wid_to_isw(fs_decode_if[f].data.wid)];
        for (genvar l = 0; l < SW; ++l) begin : g_ibuf_pop
            localparam W = l * FW + f;
            assign fs_decode_if[f].ibuf_pop[l] = is_decode_if[W % IW].ibuf_pop[W / IW];
        end
        if (SW < NW) begin : g_ibuf_pop_pad
            assign fs_decode_if[f].ibuf_pop[NW-1:SW] = '0;
        end
    end

    // issue and commit events, and per-warp drain status

    wire [NW-1:0] issued_warps;
    wire [NW-1:0] pending_alm_empty;
    for (genvar w = 0; w < NW; ++w) begin : g_warps
        assign issued_warps[w] = issue_valid[w % IW] && (issue_wis[w % IW] == ISSUE_WIS_W'(w / IW));
        assign fs_issued[w % FW][w / FW]    = issued_warps[w];
        assign fs_committed[w % FW][w / FW] = committed_warps[w];
        assign pending_alm_empty[w]         = fs_pending_alm_empty[w % FW][w / FW];
    end

    // ALU blocks: frontend slice f port q is block q * FW + f, or block f mod BA when shared

    for (genvar f = 0; f < FW; ++f) begin : g_alu_slices
        for (genvar q = 0; q < NPA; ++q) begin : g_ports
            localparam B = (BA >= FW) ? (q * FW + f) : (f % BA);
            wire own;
            if (BA >= FW) begin : g_own
                assign own = 1'b1;
            end else begin : g_shared
                assign own = (wid_to_fsw(branch_ctl_if[B].wid) == FETCH_FSW_W'(f));
            end
            assign fs_branch[f][q].valid      = branch_ctl_if[B].valid && own;
            assign fs_branch[f][q].wid        = branch_ctl_if[B].wid;
            assign fs_branch[f][q].taken      = branch_ctl_if[B].taken;
            assign fs_branch[f][q].dest       = branch_ctl_if[B].dest;
            assign fs_branch[f][q].is_trap    = branch_ctl_if[B].is_trap;
            assign fs_branch[f][q].is_mret    = branch_ctl_if[B].is_mret;
            assign fs_branch[f][q].trap_cause = branch_ctl_if[B].trap_cause;
        end
    end

    // SFU blocks: same port mapping; reads are answered by the slice owning the read wid

    for (genvar f = 0; f < FW; ++f) begin : g_sfu_slices
        for (genvar q = 0; q < NPS; ++q) begin : g_ports
            localparam B = (BS >= FW) ? (q * FW + f) : (f % BS);
            wire own_ctl, own_wr;
        `ifdef VX_CFG_EXT_RTU_ENABLE
            wire own_unlock;
        `endif
            if (BS >= FW) begin : g_own
                assign own_ctl = 1'b1;
                assign own_wr  = 1'b1;
            `ifdef VX_CFG_EXT_RTU_ENABLE
                assign own_unlock = 1'b1;
            `endif
            end else begin : g_shared
                assign own_ctl = (wid_to_fsw(warp_ctl_if[B].wid) == FETCH_FSW_W'(f));
                assign own_wr  = (wid_to_fsw(sched_csr_if[B].csr_wr_wid) == FETCH_FSW_W'(f));
            `ifdef VX_CFG_EXT_RTU_ENABLE
                assign own_unlock = (wid_to_fsw(sched_unlock_if[B].wid) == FETCH_FSW_W'(f));
            `endif
            end

            assign fs_wctl[f][q].tmc_valid   = warp_ctl_if[B].tmc_valid && own_ctl;
            assign fs_wctl[f][q].split_valid = warp_ctl_if[B].split_valid && own_ctl;
            assign fs_wctl[f][q].sjoin_valid = warp_ctl_if[B].sjoin_valid && own_ctl;
            assign fs_wctl[f][q].wsync_valid = warp_ctl_if[B].wsync_valid && own_ctl;
            assign fs_wctl[f][q].wid         = warp_ctl_if[B].wid;
            assign fs_wctl[f][q].tmc         = warp_ctl_if[B].tmc;
            assign fs_wctl[f][q].split       = warp_ctl_if[B].split;
            assign fs_wctl[f][q].sjoin       = warp_ctl_if[B].sjoin;
            assign fs_dvstack_wid[f][q]      = warp_ctl_if[B].dvstack_wid;

            assign fs_csr_rd_wid[f][q]       = sched_csr_if[B].csr_rd_wid;
            assign fs_csr_wr[f][q].valid      = sched_csr_if[B].csr_wr_valid && own_wr;
            assign fs_csr_wr[f][q].wid        = sched_csr_if[B].csr_wr_wid;
            assign fs_csr_wr[f][q].data       = sched_csr_if[B].csr_wr_data;
            assign fs_csr_wr[f][q].trap_valid = sched_csr_if[B].trap_csr_wr_valid & {NUM_TRAP_CSRS{own_wr}};
            assign fs_csr_wr[f][q].trap_data  = sched_csr_if[B].trap_csr_wr_data;

        `ifdef VX_CFG_EXT_RTU_ENABLE
            assign fs_rtu_unlock_valid[f][q] = sched_unlock_if[B].valid && own_unlock;
            assign fs_rtu_unlock_wid[f][q]   = sched_unlock_if[B].wid;
        `endif
        end
    end

    for (genvar b = 0; b < BS; ++b) begin : g_sfu_blocks
        sched_csr_rd_t csr_rd;
        if (BS >= FW) begin : g_own
            assign warp_ctl_if[b].dvstack_ptr = fs_dvstack_ptr[b % FW][b / FW];
            assign warp_ctl_if[b].mscratch    = fs_dvstack_mscratch[b % FW][b / FW];
            assign csr_rd                     = fs_csr_rd[b % FW][b / FW];
        end else begin : g_shared
            wire [FETCH_FSW_W-1:0] stack_fsw = wid_to_fsw(warp_ctl_if[b].dvstack_wid);
            wire [FETCH_FSW_W-1:0] csr_fsw   = wid_to_fsw(sched_csr_if[b].csr_rd_wid);
            assign warp_ctl_if[b].dvstack_ptr = fs_dvstack_ptr[stack_fsw][0];
            assign warp_ctl_if[b].mscratch    = fs_dvstack_mscratch[stack_fsw][0];
            assign csr_rd                     = fs_csr_rd[csr_fsw][0];
        end

        for (genvar k = 0; k < SFU_BLOCK_WARPS; ++k) begin : g_pending
            assign warp_ctl_if[b].warp_pending_alm_empty[k] = pending_alm_empty[k * BS + b];
        end
        assign warp_ctl_if[b].lsu_sched_empty = lsu_sched_empty;
        assign warp_ctl_if[b].mmu_empty       = mmu_empty;

        assign sched_csr_if[b].cycles       = cycles;
        assign sched_csr_if[b].instret      = instret;
        assign sched_csr_if[b].active_warps = active_warps;
        assign sched_csr_if[b].thread_mask  = csr_rd.thread_mask;
        assign sched_csr_if[b].mscratch     = csr_rd.mscratch;
        assign sched_csr_if[b].csr_mstatus  = csr_rd.mstatus;
        assign sched_csr_if[b].csr_mtvec    = csr_rd.mtvec;
        assign sched_csr_if[b].csr_mepc     = csr_rd.mepc;
        assign sched_csr_if[b].csr_mcause   = csr_rd.mcause;
        assign sched_csr_if[b].csr_mtval    = csr_rd.mtval;
        assign sched_csr_if[b].cta_rank     = csr_rd.cta_rank;
        assign sched_csr_if[b].cta_lane     = csr_rd.cta_lane;
    `ifdef VX_CFG_VM_ENABLE
        assign sched_csr_if[b].csr_satp     = satp[b];
        assign satp_wr_valid[b]             = sched_csr_if[b].satp_wr_valid;
        assign satp_wr_data[b]              = sched_csr_if[b].satp_wr_data;
    `endif
    end

endmodule
