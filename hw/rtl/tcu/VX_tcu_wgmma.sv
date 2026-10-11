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

`ifdef VX_CFG_TCU_WGMMA_ENABLE

// WGMMA orchestrator. Owns everything that exists only because WGMMA
// exists: dispatch-IF unpack into tbuf request, VX_tcu_tbuf instantiation,
// CTA lockstep gate, and WGMMA-specific perf counters.
//
// Observation-only on the dispatch path: this module does not drive
// .ready. The wrapper (VX_tcu_unit) feeds tbuf_ready_eff into tcu_core,
// which produces the actual handshake .ready going back to dispatch.
//
// tbuf_ready_eff is registered: a same-cycle ready closed a loop across every
// block (shared-bbuf representative + key match -> ready -> fire -> operand
// buffer enables). The tile buffers still allocate and refill from this
// cycle's heads, exactly as before; separately, each block's two candidates
// for next cycle's head (this head, or the entry behind it if this head fires)
// are checked against the state this cycle leaves behind, and fire only picks
// between the two results. A uop not yet visible in its block queue waits one
// cycle.

module VX_tcu_wgmma import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         BLOCK_SIZE      = `VX_CFG_NUM_TCU_BLOCKS,
    parameter         BANK_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE
                                      - $clog2(`VX_CFG_XLEN / 8)
                                      - $clog2(`VX_CFG_LMEM_NUM_BANKS)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output tcu_perf_t                                                    tcu_perf,
`endif

    // Observation of dispatch path (read-only; does not drive .ready).
    input  wire [BLOCK_SIZE-1:0]                                         exec_valid,
    input  wire [BLOCK_SIZE-1:0]                                         exec_ready,
    input  tcu_execute_t                                                 exec_data [BLOCK_SIZE],
    // The entry queued behind each head.
    input  wire [BLOCK_SIZE-1:0]                                         exec_next_valid,
    input  tcu_execute_t                                                 exec_next_data [BLOCK_SIZE],

    // Bank-parallel LMEM read port.
    VX_mem_bus_if.master                                                 tcu_lmem_if,

    // Outputs to tcu_core (consumed by the wrapper).
    output wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data,
    output wire [BLOCK_SIZE-1:0]                                         tbuf_ready_eff
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // CTA lockstep gate. is_first_uop / is_last_uop come from op_args.tcu,
    // set by VX_tcu_uops alongside fu_lock/fu_unlock.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0]                  cta_conflict;
    wire [BLOCK_SIZE-1:0]                  is_wgmma_b_w;
    wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0]  new_cta_b_w;
    wire [BLOCK_SIZE-1:0]                  exec_fire_b_w;
    wire [BLOCK_SIZE-1:0]                  is_first_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  is_last_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  is_setup_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  needs_setup_b_w;

    logic [BLOCK_SIZE-1:0][`VX_CFG_XLEN-1:0] desc_a_r;
    logic [BLOCK_SIZE-1:0][`VX_CFG_XLEN-1:0] desc_b_r;

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_inputs
        wire is_wgmma_op = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign is_setup_uop_b_w[bi] = is_wgmma_op && !exec_data[bi].header.wb;
        assign needs_setup_b_w[bi] =
            `ifdef VX_CFG_TCU_FEDP2K
                is_wgmma_op && !exec_data[bi].op_args.tcu.a_from_smem
            `ifdef VX_CFG_TCU_SPARSE_ENABLE
                && (exec_data[bi].op_type != INST_TCU_WGMMA_SP)
            `endif
            `else
                1'b0
            `endif
                ;
        assign is_wgmma_b_w[bi]     = exec_valid[bi] && is_wgmma_op && !is_setup_uop_b_w[bi];
        assign new_cta_b_w[bi]      = exec_data[bi].header.cta_id;
        assign exec_fire_b_w[bi]    = exec_valid[bi] && exec_ready[bi];
        assign is_first_uop_b_w[bi] = is_wgmma_op && exec_data[bi].op_args.tcu.is_first_uop;
        assign is_last_uop_b_w[bi]  = is_wgmma_op && exec_data[bi].op_args.tcu.is_last_uop;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            desc_a_r <= '0;
            desc_b_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
                if (exec_valid[bi] && exec_ready[bi]
                 && (is_setup_uop_b_w[bi]
                  || (is_first_uop_b_w[bi] && !needs_setup_b_w[bi]))) begin
                    if (exec_data[bi].op_args.tcu.a_from_smem)
                        desc_a_r[bi] <= exec_data[bi].rs1_data[0];
                    desc_b_r[bi] <= exec_data[bi].rs2_data[0];
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Next-cycle head candidates: [0] this head, [1] the entry behind it.
    // -----------------------------------------------------------------------
    tcu_tbuf_req_t [BLOCK_SIZE-1:0][1:0]  cand;
    wire [BLOCK_SIZE-1:0][1:0]            cand_compute;
    wire [BLOCK_SIZE-1:0][1:0][NCTA_WIDTH-1:0] cand_cta;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_cands
        // Candidate 1 only matters if this head fires, so its descriptor
        // assumes the setup/first-uop latch above takes this head's operands.
        wire latch_if_fire = is_setup_uop_b_w[bi] || (is_first_uop_b_w[bi] && !needs_setup_b_w[bi]);
        wire [`VX_CFG_XLEN-1:0] desc_a_after = (latch_if_fire && exec_data[bi].op_args.tcu.a_from_smem)
                                             ? exec_data[bi].rs1_data[0] : desc_a_r[bi];
        wire [`VX_CFG_XLEN-1:0] desc_b_after = latch_if_fire ? exec_data[bi].rs2_data[0] : desc_b_r[bi];
        for (genvar ci = 0; ci < 2; ++ci) begin : g_cand
            tcu_execute_t d;
            wire          v;
            `UNUSED_VAR (d)
            if (ci == 0) begin : g_head
                assign d = exec_data[bi];
                assign v = exec_valid[bi];
            end else begin : g_next
                assign d = exec_next_data[bi];
                assign v = exec_next_valid[bi];
            end
            wire is_wg = (d.op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (d.op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
            wire needs_setup_c =
                `ifdef VX_CFG_TCU_FEDP2K
                    is_wg && !d.op_args.tcu.a_from_smem
                `ifdef VX_CFG_TCU_SPARSE_ENABLE
                    && (d.op_type != INST_TCU_WGMMA_SP)
                `endif
                `else
                    1'b0
                `endif
                    ;
            wire use_live = is_wg && d.op_args.tcu.is_first_uop && !needs_setup_c;
            assign cand_compute[bi][ci] = v && is_wg && d.header.wb;
            assign cand_cta[bi][ci]     = d.header.cta_id;
            assign cand[bi][ci].valid        = cand_compute[bi][ci];
            assign cand[bi][ci].uuid         = d.header.uuid;
            assign cand[bi][ci].wid          = d.header.wid;
            assign cand[bi][ci].cta_id       = d.header.cta_id;
            assign cand[bi][ci].step_m       = d.op_args.tcu.step_m;
            assign cand[bi][ci].step_k       = d.op_args.tcu.step_k;
            assign cand[bi][ci].step_n       = d.op_args.tcu.step_n;
            assign cand[bi][ci].cd_nregs     = d.op_args.tcu.cd_nregs;
            assign cand[bi][ci].desc_a       = use_live ? d.rs1_data[0] : ((ci == 0) ? desc_a_r[bi] : desc_a_after);
            assign cand[bi][ci].desc_b       = use_live ? d.rs2_data[0] : ((ci == 0) ? desc_b_r[bi] : desc_b_after);
            assign cand[bi][ci].a_is_smem    = d.op_args.tcu.a_from_smem;
            assign cand[bi][ci].is_first_uop = d.op_args.tcu.is_first_uop;
            assign cand[bi][ci].is_last_uop  = d.op_args.tcu.is_last_uop;
            assign cand[bi][ci].setup_fire   = 1'b0;
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            assign cand[bi][ci].is_sparse    = (d.op_type == INST_TCU_WGMMA_SP);
        `endif
        end
    end

    wire [BLOCK_SIZE-1:0][1:0] cta_conflict_c;

    VX_tcu_lockstep #(
        .INSTANCE_ID (`SFORMATF(("%s-lockstep", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) lockstep (
        .clk            (clk),
        .reset          (reset),
        .is_wgmma_b     (is_wgmma_b_w),
        .new_cta_b      (new_cta_b_w),
        .exec_fire_b    (exec_fire_b_w),
        .is_first_uop_b (is_first_uop_b_w),
        .is_last_uop_b  (is_last_uop_b_w),
        .is_wgmma_c     (cand_compute),
        .new_cta_c      (cand_cta),
        .cta_conflict   (cta_conflict),
        .cta_conflict_c (cta_conflict_c)
    );

    // -----------------------------------------------------------------------
    // Per-block uop observation → VX_tcu_tbuf request.
    // req[b].valid is masked by cta_conflict so the shared bbuf doesn't
    // refill its bank-row for a CTA whose fire is currently gated by
    // lockstep.
    // -----------------------------------------------------------------------
    tcu_tbuf_req_t [BLOCK_SIZE-1:0] req;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_inputs
        wire is_wgmma_b = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        wire is_setup_uop = is_wgmma_b && !exec_data[bi].header.wb;
        assign req[bi].valid        = exec_valid[bi] && is_wgmma_b && !is_setup_uop
                                    && !cta_conflict[bi];
        assign req[bi].uuid         = exec_data[bi].header.uuid;
        assign req[bi].wid          = exec_data[bi].header.wid;
        assign req[bi].cta_id       = exec_data[bi].header.cta_id;
        assign req[bi].step_m       = exec_data[bi].op_args.tcu.step_m;
        assign req[bi].step_k       = exec_data[bi].op_args.tcu.step_k;
        assign req[bi].step_n       = exec_data[bi].op_args.tcu.step_n;
        assign req[bi].cd_nregs     = exec_data[bi].op_args.tcu.cd_nregs;
        wire use_live_desc = is_first_uop_b_w[bi]
                          && !needs_setup_b_w[bi];
        assign req[bi].desc_a       = use_live_desc ? exec_data[bi].rs1_data[0] : desc_a_r[bi];
        assign req[bi].desc_b       = use_live_desc ? exec_data[bi].rs2_data[0] : desc_b_r[bi];
        assign req[bi].a_is_smem    = exec_data[bi].op_args.tcu.a_from_smem;
        assign req[bi].is_first_uop = exec_data[bi].op_args.tcu.is_first_uop;
        assign req[bi].is_last_uop  = exec_data[bi].op_args.tcu.is_last_uop;
        assign req[bi].setup_fire   = exec_valid[bi] && exec_ready[bi] && is_setup_uop;
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        assign req[bi].is_sparse = (exec_data[bi].op_type == INST_TCU_WGMMA_SP);
    `endif
    end

    wire [BLOCK_SIZE-1:0] tbuf_ready;
    wire [BLOCK_SIZE-1:0][1:0] cand_ready_next;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] tbuf_stalls_w;
    wire [PERF_CTR_BITS-1:0] tbuf_cache_hits_w;
    wire [PERF_CTR_BITS-1:0] lmem_reads_w;
`endif

    VX_tcu_tbuf #(
        .INSTANCE_ID    (`SFORMATF(("%s-tbuf", INSTANCE_ID))),
        .NUM_BANKS      (`VX_CFG_LMEM_NUM_BANKS),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .BLOCK_SIZE     (BLOCK_SIZE)
    ) tbuf (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tbuf_stalls    (tbuf_stalls_w),
        .tbuf_cache_hits(tbuf_cache_hits_w),
        .lmem_reads     (lmem_reads_w),
    `endif
        .req            (req),
        .cand           (cand),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready     (tbuf_ready),
        .cand_ready_next(cand_ready_next)
    );
    `UNUSED_VAR (tbuf_ready)

    // Registered compute-uop ready for next cycle's head. Setup uops never
    // wait on the buffers; tcu_core passes them through on its own.
    reg [BLOCK_SIZE-1:0] tbuf_ready_r;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_eff
        wire [1:0] ok = cand_compute[bi] & cand_ready_next[bi] & ~cta_conflict_c[bi];
        always_ff @(posedge clk) begin
            if (reset) begin
                tbuf_ready_r[bi] <= 1'b0;
            end else begin
                tbuf_ready_r[bi] <= exec_fire_b_w[bi] ? ok[1] : ok[0];
            end
        end
    end
    assign tbuf_ready_eff = tbuf_ready_r;

    // -----------------------------------------------------------------------
    // WGMMA perf counters (instrs / stalls).
    // -----------------------------------------------------------------------
`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = tbuf_stalls_w;
    assign tcu_perf.tbuf_cache_hits = tbuf_cache_hits_w;
    assign tcu_perf.lmem_reads      = lmem_reads_w;

    logic wgmma_fire_b  [BLOCK_SIZE];
    logic wgmma_stall_b [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wgmma_perf
        wire is_wgmma_p = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign wgmma_fire_b [bi] = exec_valid[bi] && exec_ready[bi] && is_wgmma_p;
        assign wgmma_stall_b[bi] = exec_valid[bi] && !exec_ready[bi] && is_wgmma_p;
    end

    logic [PERF_CTR_BITS-1:0] wgmma_instrs_ctr_r;
    logic [PERF_CTR_BITS-1:0] wgmma_stalls_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            wgmma_instrs_ctr_r <= '0;
            wgmma_stalls_ctr_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
                if (wgmma_fire_b[bi])  wgmma_instrs_ctr_r <= wgmma_instrs_ctr_r + PERF_CTR_BITS'(1);
                if (wgmma_stall_b[bi]) wgmma_stalls_ctr_r <= wgmma_stalls_ctr_r + PERF_CTR_BITS'(1);
            end
        end
    end
    assign tcu_perf.wgmma_instrs = wgmma_instrs_ctr_r;
    assign tcu_perf.wgmma_stalls = wgmma_stalls_ctr_r;
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
