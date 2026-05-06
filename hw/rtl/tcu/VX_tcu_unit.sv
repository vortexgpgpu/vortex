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

module VX_tcu_unit import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output tcu_perf_t       tcu_perf,
`endif

`ifdef TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master     tcu_lmem_if,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `NUM_TCU_LANES;

    `STATIC_ASSERT (BLOCK_SIZE == `ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (tcu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    // -----------------------------------------------------------------------
    // WGMMA tile-buffer subsystem (TB-level wrapper)
    // -----------------------------------------------------------------------
    // VX_tcu_tbuf owns Q × abuf (per-block A) + 1 × bbuf (TB-shared B) +
    // LMEM-port arbitration (Q+1 → 1). Sparse meta path TBD (Phase 4 →
    // VX_tcu_mbuf).

`ifdef TCU_WGMMA_ENABLE
    localparam BANK_ADDR_WIDTH = `LMEM_LOG_SIZE - $clog2(`XLEN / 8) - $clog2(`LMEM_NUM_BANKS);

    // Per-block uop observation, packed for VX_tcu_tbuf (one bit/lane per block).
    wire [BLOCK_SIZE-1:0]                    req_valid_arr;
    wire [BLOCK_SIZE-1:0][NW_WIDTH-1:0]      req_wid_arr;
    wire [BLOCK_SIZE-1:0][3:0]               req_step_m_arr;
    wire [BLOCK_SIZE-1:0][3:0]               req_step_k_arr;
    wire [BLOCK_SIZE-1:0][3:0]               req_step_n_arr;
    wire [BLOCK_SIZE-1:0][1:0]               req_cd_nregs_arr;
    wire [BLOCK_SIZE-1:0][`XLEN-1:0]         req_desc_a_arr;
    wire [BLOCK_SIZE-1:0][`XLEN-1:0]         req_desc_b_arr;
    wire [BLOCK_SIZE-1:0]                    req_a_is_smem_arr;
`ifdef TCU_SPARSE_ENABLE
    wire [BLOCK_SIZE-1:0]                    req_is_sparse_arr;
    wire [BLOCK_SIZE-1:0][3:0]               req_fmt_s_arr;
`endif

    // cta_conflict is declared further below; forward-reference it so we can
    // mask req_valid_arr → tbuf, preventing the shared bbuf from refilling
    // its bank-row for a CTA whose fire is currently gated by lockstep.
    wire [BLOCK_SIZE-1:0] cta_conflict;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_inputs
        wire is_wgmma_b = (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        assign req_valid_arr    [bi] = per_block_execute_if[bi].valid && is_wgmma_b && !cta_conflict[bi];
        assign req_wid_arr      [bi] = per_block_execute_if[bi].data.header.wid;
        assign req_step_m_arr   [bi] = per_block_execute_if[bi].data.op_args.tcu.step_m;
        assign req_step_k_arr   [bi] = per_block_execute_if[bi].data.op_args.tcu.step_k;
        assign req_step_n_arr   [bi] = per_block_execute_if[bi].data.op_args.tcu.step_n;
        assign req_cd_nregs_arr [bi] = per_block_execute_if[bi].data.op_args.tcu.cd_nregs;
        assign req_desc_a_arr   [bi] = per_block_execute_if[bi].data.rs1_data[0];
        assign req_desc_b_arr   [bi] = per_block_execute_if[bi].data.rs2_data[0];
        assign req_a_is_smem_arr[bi] = per_block_execute_if[bi].data.op_args.tcu.a_from_smem;
    `ifdef TCU_SPARSE_ENABLE
        assign req_is_sparse_arr[bi] = per_block_execute_if[bi].data.op_args.tcu.is_sparse;
        assign req_fmt_s_arr    [bi] = per_block_execute_if[bi].data.op_args.tcu.fmt_s;
    `endif
    end

    // Per-block tile buffer outputs (rs2 broadcast, rs1 per-block, ready ANDed)
    wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`XLEN-1:0]    tbuf_rs1_data;
    wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data;
    wire [BLOCK_SIZE-1:0]                                  tbuf_ready;
`ifdef TCU_SPARSE_ENABLE
    wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0]    tbuf_sp_meta;
`endif

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] tbuf_stalls_w;
    wire [PERF_CTR_BITS-1:0] tbuf_cache_hits_w;
    wire [PERF_CTR_BITS-1:0] lmem_reads_w;
`endif

    VX_tcu_tbuf #(
        .INSTANCE_ID    (`SFORMATF(("%s-tbuf", INSTANCE_ID))),
        .NUM_BANKS      (`LMEM_NUM_BANKS),
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
        .req_valid      (req_valid_arr),
        .req_wid        (req_wid_arr),
        .req_step_m     (req_step_m_arr),
        .req_step_k     (req_step_k_arr),
        .req_step_n     (req_step_n_arr),
        .req_cd_nregs   (req_cd_nregs_arr),
        .req_desc_a     (req_desc_a_arr),
        .req_desc_b     (req_desc_b_arr),
        .req_a_is_smem  (req_a_is_smem_arr),
    `ifdef TCU_SPARSE_ENABLE
        .req_is_sparse  (req_is_sparse_arr),
        .req_fmt_s      (req_fmt_s_arr),
    `endif
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
    `ifdef TCU_SPARSE_ENABLE
        .tbuf_sp_meta   (tbuf_sp_meta),
    `endif
        .tbuf_ready     (tbuf_ready)
    );

    // -------------------------------------------------------------------
    // wgmma_instrs / wgmma_stalls: derived from per_block_execute_if.
    // -------------------------------------------------------------------

`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = tbuf_stalls_w;
    assign tcu_perf.tbuf_cache_hits = tbuf_cache_hits_w;
    assign tcu_perf.lmem_reads      = lmem_reads_w;

    logic wgmma_fire_b  [BLOCK_SIZE];
    logic wgmma_stall_b [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wgmma_perf
        wire is_wgmma_p = (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        assign wgmma_fire_b [bi] = per_block_execute_if[bi].valid && per_block_execute_if[bi].ready && is_wgmma_p;
        assign wgmma_stall_b[bi] = per_block_execute_if[bi].valid && !per_block_execute_if[bi].ready && is_wgmma_p;
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

`else // !TCU_WGMMA_ENABLE

`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = '0;
    assign tcu_perf.tbuf_cache_hits = '0;
    assign tcu_perf.lmem_reads      = '0;
    assign tcu_perf.wgmma_instrs    = '0;
    assign tcu_perf.wgmma_stalls    = '0;
`endif

`endif // TCU_WGMMA_ENABLE

    // -----------------------------------------------------------------------
    // CTA lockstep gate
    //
    // Mirrors SimX's CTA-overlap fence (sim/simx/tcu/tcu_unit.cpp): the shared
    // tbuf assumes single-CTA occupancy across all blocks. A new WGMMA on
    // block b is deferred while any other block holds an in-flight uop with
    // a different cta_id, so that all warps of the current CTA drain before
    // a new CTA enters. Same-CTA across blocks (the production case for a
    // warpgroup at the same uop) remains free.
    //
    // A small per-block uop counter (in_flight_count) tracks accept→drain.
    // Counting all TCU types is conservative: a non-WGMMA op (WMMA / META)
    // mid-flight will also block a different CTA's WGMMA from entering, even
    // though it doesn't share state. Acceptable — favours correctness over
    // throughput on a path that is already config-rare.
    // -----------------------------------------------------------------------

`ifdef TCU_WGMMA_ENABLE
    localparam INFLIGHT_CW = 4;
    reg [BLOCK_SIZE-1:0][INFLIGHT_CW-1:0] inflight_count_r;
    reg [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0]  cta_owner_r;
    // Per-block "WGMMA expansion in progress" — set on first sub-uop fire,
    // cleared on last sub-uop fire. Persists across LMEM-stall gaps so the
    // lockstep gate keeps a different CTA's WGMMA from sneaking in mid-
    // expansion and corrupting the shared bbuf descriptor latch.
    reg [BLOCK_SIZE-1:0]                  in_expansion_r;

    wire [BLOCK_SIZE-1:0] block_in_flight;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_block_in_flight
        assign block_in_flight[bi] = (inflight_count_r[bi] != '0) || in_expansion_r[bi];
    end

    // Per-block fire intent (raw — without lockstep mask). is_wgmma_b is the
    // gate trigger; tbuf_ready_raw is the unmasked downstream readiness.
    wire [BLOCK_SIZE-1:0]                  is_wgmma_b_w;
    wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0]  new_cta_b_w;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_intent
        assign is_wgmma_b_w[bi] = per_block_execute_if[bi].valid
                              && (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        assign new_cta_b_w[bi] = per_block_execute_if[bi].data.header.cta_id;
    end

    // Sequential propagation in priority order. Each block's "active" state
    // includes:
    //   1. Registered in-flight (uops mid-pipeline).
    //   2. Queued WGMMA (per_block_execute_if.valid). This is critical —
    //      between uops of the same warp's expansion (and at iteration
    //      boundaries), inflight_count_r briefly drops to 0 even though the
    //      block is logically still mid-WGMMA. Without considering the
    //      queued op, another CTA could sneak in and the bbuf would refill
    //      mid-expansion. Mirrors SimX's wgmma_planned_warps_ semantic.
    //   3. This-cycle claim from lower-index blocks (priority arbitration so
    //      two blocks queued with different CTAs don't both think they win).
    logic [BLOCK_SIZE-1:0]                 eff_in_flight;
    logic [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] eff_cta_owner;
    logic [BLOCK_SIZE-1:0]                 cta_conflict_int;
    assign cta_conflict = cta_conflict_int;
    always_comb begin
        for (int b = 0; b < BLOCK_SIZE; b++) begin
            eff_in_flight[b] = block_in_flight[b];
            eff_cta_owner[b] = cta_owner_r[b];
            cta_conflict_int[b] = 1'b0;
        end
        for (int b = 0; b < BLOCK_SIZE; b++) begin
            logic any_other_diff;
            any_other_diff = 1'b0;
            for (int k = 0; k < BLOCK_SIZE; k++) begin
                if (k != b && eff_in_flight[k] && (eff_cta_owner[k] != new_cta_b_w[b]))
                    any_other_diff = 1'b1;
            end
            cta_conflict_int[b] = is_wgmma_b_w[b] && any_other_diff;

            // Claim if block has a queued WGMMA and no conflict — regardless
            // of tbuf_ready. The claim represents lockstep ownership, not
            // immediate fire. Subsequent (higher-index) blocks observe this
            // block as active and check against its cta.
            if (is_wgmma_b_w[b] && !any_other_diff) begin
                eff_in_flight[b] = 1'b1;
                eff_cta_owner[b] = new_cta_b_w[b];
            end
        end
    end

    wire [BLOCK_SIZE-1:0] tbuf_ready_eff;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_eff
        assign tbuf_ready_eff[bi] = tbuf_ready[bi] && !cta_conflict[bi];
    end

    // State updates: count execute_if fires (in) vs result_if fires (out).
    // Latch cta_owner on the 0→1 transition (first uop entering an idle block).
    // in_expansion_r persists from first sub-uop (step_{m,n,k}=0) to last sub-
    // uop (step_k=1, step_m=1, max step_n for cd_nregs) so the gate stays
    // engaged across LMEM-stall gaps within a single WGMMA expansion.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_inflight_upd
        wire exec_fire_b   = per_block_execute_if[bi].valid && per_block_execute_if[bi].ready;
        wire result_fire_b = per_block_result_if[bi].valid && per_block_result_if[bi].ready;
        wire is_wgmma_fire = exec_fire_b && (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        wire [3:0] sm = per_block_execute_if[bi].data.op_args.tcu.step_m;
        wire [3:0] sn = per_block_execute_if[bi].data.op_args.tcu.step_n;
        wire [3:0] sk = per_block_execute_if[bi].data.op_args.tcu.step_k;
        wire [1:0] cdn = per_block_execute_if[bi].data.op_args.tcu.cd_nregs;
        wire is_first_uop = (sm == 4'd0) && (sn == 4'd0) && (sk == 4'd0);
        wire [3:0] last_sn = (cdn == 2'd0) ? 4'd3
                           : (cdn == 2'd1) ? 4'd7
                                           : 4'd15;
        wire is_last_uop  = (sm == 4'd1) && (sk == 4'd1) && (sn == last_sn);
        wire [NCTA_WIDTH-1:0] new_cta_b = per_block_execute_if[bi].data.header.cta_id;
        always @(posedge clk) begin
            if (reset) begin
                inflight_count_r[bi] <= '0;
                cta_owner_r[bi]      <= '0;
                in_expansion_r[bi]   <= 1'b0;
            end else begin
                case ({exec_fire_b, result_fire_b})
                    2'b10: inflight_count_r[bi] <= inflight_count_r[bi] + INFLIGHT_CW'(1);
                    2'b01: inflight_count_r[bi] <= inflight_count_r[bi] - INFLIGHT_CW'(1);
                    default: ;
                endcase
                if (exec_fire_b && (inflight_count_r[bi] == '0) && !in_expansion_r[bi])
                    cta_owner_r[bi] <= new_cta_b;
                if (is_wgmma_fire) begin
                    if (is_first_uop) in_expansion_r[bi] <= 1'b1;
                    if (is_last_uop)  in_expansion_r[bi] <= 1'b0;
                end
            end
        end
    end

    // -------------------------------------------------------------------
    // Sim-only invariant: post-gate, a fired WGMMA on block b must agree
    // with every other block's effective cta_owner (registered + this-cycle
    // claims from blocks with index < b). Catches gate bypass during
    // simulation; mirrors the SimX runtime check in tcu_unit.cpp.
    // -------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] lockstep_violation;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_violation
        wire exec_fire_b = per_block_execute_if[bi].valid && per_block_execute_if[bi].ready
                        && (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        // exec_fire_b implies cta_conflict[bi]==0; the assertion checks that
        // none of the OTHER blocks' effective owners disagree at fire time.
        assign lockstep_violation[bi] = exec_fire_b && cta_conflict[bi];
    end
    `RUNTIME_ASSERT (~|lockstep_violation,
        ("%s lockstep violation: a WGMMA uop fired with a cta_id that conflicts with another block's resident CTA",
         INSTANCE_ID))
`endif // TCU_WGMMA_ENABLE

    // -----------------------------------------------------------------------
    // TCU core instances
    // -----------------------------------------------------------------------

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        VX_tcu_core #(
            .INSTANCE_ID (`SFORMATF(("%s-fused%0d", INSTANCE_ID, block_idx)))
        ) tcu_core (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
        `ifdef TCU_WGMMA_ENABLE
            .tbuf_rs1_data (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data (tbuf_rs2_data[block_idx]),
        `ifdef TCU_SPARSE_ENABLE
            .tbuf_sp_meta  (tbuf_sp_meta[block_idx]),
        `endif
            .tbuf_ready    (tbuf_ready_eff[block_idx]),
        `endif
            .execute_if (per_block_execute_if[block_idx]),
            .result_if  (per_block_result_if[block_idx])
        );
    end

    // -----------------------------------------------------------------------
    // Lane gather
    // -----------------------------------------------------------------------

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
