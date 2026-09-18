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

`ifdef TCU_META_ENABLE
    // TCU_LD memory client connection to VX_lsu_scheduler at VX_core.
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

`ifdef TCU_OP
    VX_lsu_mem_if.master    tcu_lsu_mem_if,
    VX_txbar_bus_if.master  txbar_bus_if,
`endif

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

`ifndef TCU_OP
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

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_split
    `ifdef TCU_META_ENABLE
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
    // Result_if merge: AGU result and tcu_core result are mutually exclusive
    // in time per block; OR-mux with priority arbiter (AGU wins: TCU_LD is rare).
    // -----------------------------------------------------------------------
`ifdef TCU_META_ENABLE
    // AGU wins same-cycle conflicts; tcu_core stalls (ready=0) and retries next cycle.
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

    // -----------------------------------------------------------------------
    // TCU core instances
    // -----------------------------------------------------------------------

`else // TCU_OP

`ifdef PERF_ENABLE
    assign tcu_perf = '0;
`endif

    VX_txbar_bus_if per_block_txbar_if[BLOCK_SIZE]();

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) per_block_lsu_mem_if[BLOCK_SIZE]();

`endif // TCU_OP

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef TCU_OP
        VX_tcu_op_core #(
            .INSTANCE_ID (`SFORMATF(("%s-op_core%0d", INSTANCE_ID, block_idx)))
        ) tcu_fp (
            `SCOPE_IO_BIND (block_idx)
            .clk            (clk),
            .reset          (reset),
            .execute_if     (per_block_execute_if[block_idx]),
            .tcu_lsu_mem_if (per_block_lsu_mem_if[block_idx]),
            .txbar_bus_if   (per_block_txbar_if[block_idx]),
            .result_if      (per_block_result_if[block_idx])
        );
    `else
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
        `ifdef TCU_META_ENABLE
            .ext_meta_wr_en   (agu_meta_wr_en),
            .ext_meta_wr_wid  (agu_meta_wr_wid),
            .ext_meta_wr_idx  (agu_meta_wr_idx),
            .ext_meta_wr_data (agu_meta_wr_data),
        `endif
            .execute_if (core_execute_if[block_idx]),
            .result_if  (core_result_if[block_idx])
        );
    `endif
    end

`ifdef TCU_OP
    VX_txbar_arb #(
        .NUM_REQS (BLOCK_SIZE),
        .ARBITER  ("R"),
        .OUT_BUF  (0)
    ) txbar_arb (
        .clk       (clk),
        .reset     (reset),
        .bus_in_if (per_block_txbar_if),
        .bus_out_if(txbar_bus_if)
    );

    // Only issue block 0 reaches memory: the TCU_OP programming model runs a
    // single self-managed warp (warp 0 => issue block 0), so the other blocks
    // never issue MMA_OPs. Binding every block to the shared tcu_lsu_mem_if
    // multi-drives it (idle blocks' zeros mask the active block's requests);
    // dead-ending the idle blocks makes a stray MMA_OP on one of them
    // back-pressure forever instead of silently corrupting the bus.
    assign tcu_lsu_mem_if.req_valid = per_block_lsu_mem_if[0].req_valid;
    assign tcu_lsu_mem_if.req_data  = per_block_lsu_mem_if[0].req_data;
    assign per_block_lsu_mem_if[0].req_ready = tcu_lsu_mem_if.req_ready;
    assign per_block_lsu_mem_if[0].rsp_valid = tcu_lsu_mem_if.rsp_valid;
    assign per_block_lsu_mem_if[0].rsp_data  = tcu_lsu_mem_if.rsp_data;
    assign tcu_lsu_mem_if.rsp_ready = per_block_lsu_mem_if[0].rsp_ready;

    for (genvar block_idx = 1; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tcu_mem_tieoff
        assign per_block_lsu_mem_if[block_idx].req_ready = 1'b0;
        assign per_block_lsu_mem_if[block_idx].rsp_valid = 1'b0;
        assign per_block_lsu_mem_if[block_idx].rsp_data  = '0;
        // "Only block 0 issues" is an assumption about the warp-to-issue-block
        // mapping, not an invariant the hardware enforces. If an MMA_OP ever
        // lands on another block it back-pressures forever, which presents as
        // an unexplained hang. Name it instead.
        `RUNTIME_ASSERT(~per_block_lsu_mem_if[block_idx].req_valid,
            ("%t: *** %s: MMA_OP issued on TCU block %0d, but only block 0 is connected to memory; this would hang. Check the warp-to-issue-block mapping of the MMA-issuing warp.", $time, INSTANCE_ID, block_idx))
        `UNUSED_VAR (per_block_lsu_mem_if[block_idx].req_valid)
        `UNUSED_VAR (per_block_lsu_mem_if[block_idx].req_data)
        `UNUSED_VAR (per_block_lsu_mem_if[block_idx].rsp_ready)
    end
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

    // Debugging
    always_ff @(posedge clk) begin
        if (~reset && per_block_execute_if[0].valid && per_block_execute_if[0].ready) begin
        `ifdef TCU_OP
            if (per_block_execute_if[0].data.op_type == INST_TCU_MMA_OP) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (outer-product)\n", $time));
            end
        `else
            if (per_block_execute_if[0].data.op_type == INST_TCU_WMMA) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (inner-product dense)\n", $time)); 
            end
            `ifdef TCU_SPARSE_ENABLE
            else if (per_block_execute_if[0].data.op_type == INST_TCU_WMMA_SP) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (inner-product sparse)\n", $time));
            end
            `endif
        `endif
        end
    end

endmodule
