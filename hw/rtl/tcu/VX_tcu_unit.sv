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
`ifdef VX_CFG_TCU_TMEM_ENABLE
    output tmem_perf_t      tmem_perf,
`endif
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master     tcu_lmem_if,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory client connection to VX_lsu_scheduler at VX_core.
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    input wire [`VX_CFG_NUM_WARPS-1:0][NW_WIDTH-1:0] cta_rank_table,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_TCU_LANES;

    `STATIC_ASSERT (BLOCK_SIZE == `VX_CFG_ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_THREADS, ("must be full warp execution"));
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
    // Split each per_block_execute_if between two consumers:
    //   - VX_tcu_agu: handles INST_TCU_LD (warp-level memory load).
    //   - VX_tcu_core: handles every MMA op_type.
    // The ready signal is muxed by op_type so only one consumer drives at a time.
    // -----------------------------------------------------------------------
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) core_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (tcu_result_t)
    ) core_result_if[BLOCK_SIZE]();

`ifdef TCU_META_ENABLE
    wire [BLOCK_SIZE-1:0]    agu_ld_valid;
    wire [BLOCK_SIZE-1:0]    agu_ld_ready;
    tcu_execute_t            agu_ld_data [BLOCK_SIZE];

    wire [BLOCK_SIZE-1:0]    agu_result_valid;
    tcu_result_t             agu_result_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]    agu_result_ready;
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // TMEM_ALLOC/DEALLOC/ST/LD bypass tcu_core entirely (no FEDP compute)
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) tmem_execute_if[BLOCK_SIZE]();
`endif

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_split
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        wire is_tmem_mgmt = (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_ALLOC)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_DEALLOC)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_ST)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_LD);
        assign tmem_execute_if[bi].valid = per_block_execute_if[bi].valid && is_tmem_mgmt;
        assign tmem_execute_if[bi].data  = per_block_execute_if[bi].data;
    `endif
    `ifdef TCU_META_ENABLE
        wire is_tcu_ld = (per_block_execute_if[bi].data.op_type == INST_TCU_LD);

        // To AGU when TCU_LD
        assign agu_ld_valid[bi]    = per_block_execute_if[bi].valid && is_tcu_ld;
        assign agu_ld_data[bi]     = per_block_execute_if[bi].data;

        // To tcu_core when NOT TCU_LD (and not TMEM management, if enabled)
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid && !is_tcu_ld
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            && !is_tmem_mgmt
        `endif
            ;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;

        // Parent .ready: route to AGU on TCU_LD, TMEM bypass on TMEM
        // management, otherwise tcu_core
        assign per_block_execute_if[bi].ready = is_tcu_ld
            ? agu_ld_ready[bi]
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            : is_tmem_mgmt
            ? tmem_execute_if[bi].ready
        `endif
            : core_execute_if[bi].ready;
    `else
        // No sparse: pass-through to tcu_core (or TMEM bypass, if enabled)
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            && !is_tmem_mgmt
        `endif
            ;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;
        assign per_block_execute_if[bi].ready =
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            is_tmem_mgmt ? tmem_execute_if[bi].ready :
        `endif
            core_execute_if[bi].ready;
    `endif
    end

`ifdef VX_CFG_TCU_TMEM_ENABLE
    VX_result_if #(
        .data_t (tcu_result_t)
    ) tmem_result_if[BLOCK_SIZE]();
`endif

    // -----------------------------------------------------------------------
    // Result_if merge: AGU / TMEM-bypass / tcu_core results are mutually
    // exclusive in time per block; OR-mux with priority arbiter (AGU wins,
    // then TMEM bypass: both are rare single-cycle ops vs. tcu_core's
    // multi-cycle FEDP compute).
    // -----------------------------------------------------------------------
`ifdef TCU_META_ENABLE
    // AGU wins same-cycle conflicts; tcu_core stalls (ready=0) and retries next cycle.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_merge
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        assign per_block_result_if[bi].valid = agu_result_valid[bi] || tmem_result_if[bi].valid || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = agu_result_valid[bi]  ? agu_result_data[bi]
                                              : tmem_result_if[bi].valid ? tmem_result_if[bi].data
                                              : core_result_if[bi].data;
        assign agu_result_ready[bi]      = per_block_result_if[bi].ready;
        assign tmem_result_if[bi].ready  = per_block_result_if[bi].ready && !agu_result_valid[bi];
        assign core_result_if[bi].ready  = per_block_result_if[bi].ready && !agu_result_valid[bi] && !tmem_result_if[bi].valid;
    `else
        assign per_block_result_if[bi].valid = agu_result_valid[bi] || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = agu_result_valid[bi]
            ? agu_result_data[bi]
            : core_result_if[bi].data;
        assign agu_result_ready[bi]    = per_block_result_if[bi].ready;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !agu_result_valid[bi];
    `endif
    end
`else
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_passthru
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        assign per_block_result_if[bi].valid = tmem_result_if[bi].valid || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = tmem_result_if[bi].valid ? tmem_result_if[bi].data : core_result_if[bi].data;
        assign tmem_result_if[bi].ready = per_block_result_if[bi].ready;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !tmem_result_if[bi].valid;
    `else
        assign per_block_result_if[bi].valid = core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = core_result_if[bi].data;
        assign core_result_if[bi].ready      = per_block_result_if[bi].ready;
    `endif
    end
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // abuf A-stripe fill from TMEM (UMMA TS form). A's rows are the CTA's whole
    // lane range, so the lane base is the bank origin for the block's CTA rank.
    wire [BLOCK_SIZE-1:0]                        afill_valid;
    wire [TCU_TMEM_COL_BITS-1:0]                 afill_col_base  [BLOCK_SIZE];
    wire [TCU_TMEM_LANE_BITS-1:0]                afill_lane_base [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]                        afill_grant;
    wire [TCU_WG_TILE_M-1:0][TCU_TC_N-1:0][31:0] afill_data      [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_afill_lane
        assign afill_lane_base[bi] = 
            TCU_TMEM_LANE_BITS'(cta_rank_table[core_execute_if[bi].data.header.wid]) *
            TCU_TMEM_LANE_BITS'(TCU_WG_TILE_M);
    end
`endif

    // -----------------------------------------------------------------------
    // WGMMA feature (orchestrator): VX_tcu_tbuf + VX_tcu_lockstep + perf.
    // -----------------------------------------------------------------------

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [BLOCK_SIZE-1:0]                                          exec_valid_w;
    wire [BLOCK_SIZE-1:0]                                          exec_ready_w;
    tcu_execute_t                                                  exec_data_w [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_obs
        assign exec_valid_w[bi]    = core_execute_if[bi].valid;
        assign exec_ready_w[bi]    = core_execute_if[bi].ready;
        assign exec_data_w[bi]     = core_execute_if[bi].data;
    end

    wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data;
    wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data;
    wire [BLOCK_SIZE-1:0]                                         tbuf_ready_eff;

    VX_tcu_wgmma #(
        .INSTANCE_ID (`SFORMATF(("%s-wgmma", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) wgmma (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
        .exec_valid     (exec_valid_w),
        .exec_ready     (exec_ready_w),
        .exec_data      (exec_data_w),
        .tcu_lmem_if    (tcu_lmem_if),
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        .afill_valid    (afill_valid),
        .afill_col_base (afill_col_base),
        .afill_grant    (afill_grant),
        .afill_data     (afill_data),
    `endif
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready_eff (tbuf_ready_eff)
    );

`else // !VX_CFG_TCU_WGMMA_ENABLE

`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = '0;
    assign tcu_perf.tbuf_cache_hits = '0;
    assign tcu_perf.lmem_reads      = '0;
    assign tcu_perf.wgmma_instrs    = '0;
    assign tcu_perf.wgmma_stalls    = '0;
`endif

`endif // VX_CFG_TCU_WGMMA_ENABLE

    // -----------------------------------------------------------------------
    // VX_tcu_agu — warp-level AGU for TCU_LD instructions.
    // Drives meta_wr signals broadcast to every tcu_core so wmma_sp on
    // any block sees the loaded metadata.
    // -----------------------------------------------------------------------
`ifdef TCU_META_ENABLE
    // The AGU streams NUM_TCU_LANES operands over the LSU memory client, whose
    // request/response mask and data are sized by NUM_LSU_LANES; the two lane
    // counts must match for the per-lane completion tracking to be correct.
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_LSU_LANES, ("VX_tcu_agu requires NUM_TCU_LANES == NUM_LSU_LANES"));

    wire                                              agu_meta_wr_en;
    wire [NW_WIDTH-1:0]                               agu_meta_wr_wid;
    wire [4:0]                                        agu_meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]        agu_meta_wr_data;

    VX_tcu_agu #(
        .INSTANCE_ID (`SFORMATF(("%s-agu", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE),
        .NUM_LANES   (NUM_LANES)
    ) agu (
        .clk                (clk),
        .reset              (reset),
        .per_block_ld_valid (agu_ld_valid),
        .per_block_ld_data  (agu_ld_data),
        .per_block_ld_ready (agu_ld_ready),
        .client_if          (tcu_mem_if),
        .meta_wr_en         (agu_meta_wr_en),
        .meta_wr_wid        (agu_meta_wr_wid),
        .meta_wr_idx        (agu_meta_wr_idx),
        .meta_wr_data       (agu_meta_wr_data),
        .result_valid       (agu_result_valid),
        .result_data        (agu_result_data),
        .result_ready       (agu_result_ready)
    );
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // -----------------------------------------------------------------------
    // TMEM storage/management (VX_tcu_tmem) + its wiring to the per-block
    // tcu_core instances
    // -----------------------------------------------------------------------

    wire        [BLOCK_SIZE-1:0] tmem_mgmt_valid;
    tcu_execute_t                tmem_mgmt_data [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_extract
        assign tmem_mgmt_valid[bi] = tmem_execute_if[bi].valid;
        assign tmem_mgmt_data[bi]  = tmem_execute_if[bi].data;
    end

    wire        [BLOCK_SIZE-1:0] tmem_mgmt_ready;
    wire        [BLOCK_SIZE-1:0] tmem_result_valid_w;
    tcu_result_t                 tmem_result_data_w [BLOCK_SIZE];
    wire        [BLOCK_SIZE-1:0] tmem_result_ready_w;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_bridge
        assign tmem_execute_if[bi].ready = tmem_mgmt_ready[bi];
        assign tmem_result_if[bi].valid  = tmem_result_valid_w[bi];
        assign tmem_result_if[bi].data   = tmem_result_data_w[bi];
        assign tmem_result_ready_w[bi]   = tmem_result_if[bi].ready;
    end

    // UMMA compute read (operand C) / writeback ports to/from tcu_core.
    wire [BLOCK_SIZE-1:0]                   tmem_rd_valid;
    wire [TCU_TMEM_LANE_BITS-1:0]           tmem_rd_lane_base [BLOCK_SIZE];
    wire [TCU_TMEM_COL_BITS-1:0]            tmem_rd_col_base  [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]                   tmem_rd_grant;
    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] tmem_rd_data      [BLOCK_SIZE];

    wire                                    tmem_wr_valid     [BLOCK_SIZE];
    wire [TCU_TMEM_LANE_BITS-1:0]           tmem_wr_lane_base [BLOCK_SIZE];
    wire [TCU_TMEM_COL_BITS-1:0]            tmem_wr_col_base  [BLOCK_SIZE];
    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] tmem_wr_data      [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]                   tmem_wr_grant;
`ifdef PERF_ENABLE
    wire [BLOCK_SIZE-1:0]                   perf_umma_hazard_stall;
    wire [BLOCK_SIZE-1:0]                   perf_umma_rd_stall;
`endif

    VX_tcu_tmem #(
        .INSTANCE_ID (`SFORMATF(("%s-tmem", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) tmem (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_umma_instrs      (tmem_perf.umma_instrs),
        .perf_tmem_reads       (tmem_perf.tmem_reads),
        .perf_tmem_writes      (tmem_perf.tmem_writes),
        .perf_tmem_bank_stalls (tmem_perf.tmem_bank_stalls),
    `endif
        .mgmt_valid    (tmem_mgmt_valid),
        .mgmt_data     (tmem_mgmt_data),
        .mgmt_ready    (tmem_mgmt_ready),
        .result_valid  (tmem_result_valid_w),
        .result_data   (tmem_result_data_w),
        .result_ready  (tmem_result_ready_w),
        .rd_valid      (tmem_rd_valid),
        .rd_lane_base  (tmem_rd_lane_base),
        .rd_col_base   (tmem_rd_col_base),
        .rd_grant      (tmem_rd_grant),
        .rd_data       (tmem_rd_data),
        .wr_valid      (tmem_wr_valid),
        .wr_lane_base  (tmem_wr_lane_base),
        .wr_col_base   (tmem_wr_col_base),
        .wr_data       (tmem_wr_data),
        .wr_grant      (tmem_wr_grant),
        .afill_valid    (afill_valid),
        .afill_lane_base(afill_lane_base),
        .afill_col_base (afill_col_base),
        .afill_grant    (afill_grant),
        .afill_data     (afill_data)
    );
`endif // VX_CFG_TCU_TMEM_ENABLE

    // -----------------------------------------------------------------------
    // TCU core instances
    // -----------------------------------------------------------------------

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        // This block's currently-executing warp's CTA-local rank
        wire [NW_WIDTH-1:0] block_cta_rank =
            cta_rank_table[core_execute_if[block_idx].data.header.wid];
    `endif
        VX_tcu_core #(
            .INSTANCE_ID (`SFORMATF(("%s-fused%0d", INSTANCE_ID, block_idx)))
        ) tcu_core (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .tbuf_rs1_data (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data (tbuf_rs2_data[block_idx]),
            .tbuf_ready    (tbuf_ready_eff[block_idx]),
        `endif
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            .cta_rank         (block_cta_rank),
            .tmem_rd_valid    (tmem_rd_valid[block_idx]),
            .tmem_rd_lane_base(tmem_rd_lane_base[block_idx]),
            .tmem_rd_col_base (tmem_rd_col_base[block_idx]),
            .tmem_rd_grant    (tmem_rd_grant[block_idx]),
            .tmem_rd_data     (tmem_rd_data[block_idx]),
            .tmem_wr_valid    (tmem_wr_valid[block_idx]),
            .tmem_wr_lane_base(tmem_wr_lane_base[block_idx]),
            .tmem_wr_col_base (tmem_wr_col_base[block_idx]),
            .tmem_wr_data     (tmem_wr_data[block_idx]),
            .tmem_wr_grant    (tmem_wr_grant[block_idx]),
        `ifdef PERF_ENABLE
            .perf_umma_hazard_stall (perf_umma_hazard_stall[block_idx]),
            .perf_umma_rd_stall     (perf_umma_rd_stall[block_idx]),
        `endif
        `endif
        `ifdef TCU_META_ENABLE
            .ext_meta_wr_en   (agu_meta_wr_en),
            .ext_meta_wr_wid  (agu_meta_wr_wid),
            .ext_meta_wr_idx  (agu_meta_wr_idx),
            .ext_meta_wr_data (agu_meta_wr_data),
        `endif
            .execute_if (core_execute_if[block_idx]),
            .result_if  (core_result_if[block_idx])
        );
    end

`ifdef VX_CFG_TCU_TMEM_ENABLE
`ifdef PERF_ENABLE
    // RAW-interlock stalls, summed across blocks
    logic [PERF_CTR_BITS-1:0] tmem_hazard_stalls_r;
    logic [PERF_CTR_BITS-1:0] tmem_rd_grant_stalls_r;
    always @(posedge clk) begin
        if (reset) begin
            tmem_hazard_stalls_r   <= '0;
            tmem_rd_grant_stalls_r <= '0;
        end else begin
            tmem_hazard_stalls_r   <= tmem_hazard_stalls_r
                + PERF_CTR_BITS'($countones(perf_umma_hazard_stall));
            tmem_rd_grant_stalls_r <= tmem_rd_grant_stalls_r
                + PERF_CTR_BITS'($countones(perf_umma_rd_stall));
        end
    end
    assign tmem_perf.tmem_hazard_stalls   = tmem_hazard_stalls_r;
    assign tmem_perf.tmem_rd_grant_stalls = tmem_rd_grant_stalls_r;
`endif
`endif

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
