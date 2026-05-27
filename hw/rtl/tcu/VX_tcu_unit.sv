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

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master     tcu_lmem_if,
`endif

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // P2d: TCU_LD memory client connection to VX_lsu_scheduler at VX_core.
    VX_lsu_sched_if.master  tcu_mem_if,
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
    // P2d: split each per_block_execute_if between two consumers:
    //   - VX_tcu_agu: handles INST_TCU_LD (warp-level memory load).
    //   - VX_tcu_core: handles every other TCU op_type (WMMA, WGMMA, META_STORE).
    // The ready signal is muxed by op_type so only one consumer drives at a time.
    // -----------------------------------------------------------------------
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) core_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (tcu_result_t)
    ) core_result_if[BLOCK_SIZE]();

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [BLOCK_SIZE-1:0]    agu_ld_valid;
    wire [BLOCK_SIZE-1:0]    agu_ld_ready;
    tcu_execute_t            agu_ld_data [BLOCK_SIZE];

    wire [BLOCK_SIZE-1:0]    agu_result_valid;
    tcu_result_t             agu_result_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]    agu_result_ready;
`endif

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_split
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        wire is_tcu_ld = (per_block_execute_if[bi].data.op_type == INST_TCU_LD);

        // To AGU when TCU_LD
        assign agu_ld_valid[bi]    = per_block_execute_if[bi].valid && is_tcu_ld;
        assign agu_ld_data[bi]     = per_block_execute_if[bi].data;

        // To tcu_core when NOT TCU_LD
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid && !is_tcu_ld;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;

        // Parent .ready: route to AGU on TCU_LD, otherwise to tcu_core
        assign per_block_execute_if[bi].ready = is_tcu_ld
            ? agu_ld_ready[bi]
            : core_execute_if[bi].ready;
    `else
        // No sparse: pass-through to tcu_core
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;
        assign per_block_execute_if[bi].ready = core_execute_if[bi].ready;
    `endif
    end

    // -----------------------------------------------------------------------
    // P2d: result_if merge. AGU result and tcu_core result are mutually
    // exclusive in time per block, but the result_if interface only has
    // one valid/ready pair, so OR-mux with a priority arbiter.
    // Priority: AGU first (TCU_LD is rare and short).
    // -----------------------------------------------------------------------
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Priority arbiter: AGU wins when both valid same cycle. tcu_core's
    // pipeline preserves its data while ready=0, so the next cycle it
    // wins. This is the expected steady-state when TCU_LDs and WMMA_SPs
    // overlap in the warp's instruction stream.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_merge
        assign per_block_result_if[bi].valid = agu_result_valid[bi] || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = agu_result_valid[bi]
            ? agu_result_data[bi]
            : core_result_if[bi].data;
        assign agu_result_ready[bi]    = per_block_result_if[bi].ready;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !agu_result_valid[bi];
    end
`else
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_passthru
        assign per_block_result_if[bi].valid = core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = core_result_if[bi].data;
        assign core_result_if[bi].ready      = per_block_result_if[bi].ready;
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

    wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]    tbuf_rs1_data;
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
    // P2d: VX_tcu_agu — warp-level AGU for TCU_LD instructions.
    // Drives meta_wr signals broadcast to every tcu_core (so wmma_sp on
    // any block sees the loaded metadata).
    // -----------------------------------------------------------------------
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire                                              agu_meta_wr_en;
    wire [NW_WIDTH-1:0]                               agu_meta_wr_wid;
    wire [3:0]                                        agu_meta_wr_idx;
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
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .tbuf_rs1_data (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data (tbuf_rs2_data[block_idx]),
            .tbuf_ready    (tbuf_ready_eff[block_idx]),
        `endif
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            .ext_meta_wr_en   (agu_meta_wr_en),
            .ext_meta_wr_wid  (agu_meta_wr_wid),
            .ext_meta_wr_idx  (agu_meta_wr_idx),
            .ext_meta_wr_data (agu_meta_wr_data),
        `endif
            .execute_if (core_execute_if[block_idx]),
            .result_if  (core_result_if[block_idx])
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
