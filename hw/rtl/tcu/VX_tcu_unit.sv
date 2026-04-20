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
    // Centralized WGMMA tile buffer (shared across all BLOCK_SIZE cores)
    // -----------------------------------------------------------------------

`ifdef TCU_WGMMA_ENABLE
    localparam BANK_ADDR_WIDTH = `LMEM_LOG_SIZE - $clog2(`XLEN / 8) - $clog2(`LMEM_NUM_BANKS);

    // Shared tile buffer outputs — broadcast to all BLOCK_SIZE cores
    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0]    shared_tbuf_rs1_data;
    wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] shared_tbuf_rs2_data;
`ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0]     shared_tbuf_sp_meta;
`endif
    wire                                    shared_tbuf_ready;

    // All warps are synchronized via SW barrier (__syncthreads) and execute
    // the same uop sequence in lockstep. Use block 0's control signals to
    // drive the single shared tile buffer.
    wire is_wgmma_shared = (per_block_execute_if[0].data.op_type == INST_TCU_WGMMA);

    wire req_valid_shared = per_block_execute_if[0].valid && is_wgmma_shared;
    wire req_fire_shared  = per_block_execute_if[0].valid
                         && per_block_execute_if[0].ready
                         && is_wgmma_shared;

    // Single LMEM interface — no per-block arbiter needed
    VX_mem_bus_if #(
        .DATA_SIZE  (`LMEM_NUM_BANKS * (`XLEN / 8)),
        .TAG_WIDTH  (TCU_LMEM_TAG_W),
        .FLAGS_WIDTH(LMEM_DMA_FLAGS_W),
        .ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) tbuf_lmem_if();

    VX_tcu_tbuf #(
        .INSTANCE_ID    (`SFORMATF(("%s-tbuf", INSTANCE_ID))),
        .NUM_BANKS      (`LMEM_NUM_BANKS),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) shared_tile_buf (
        .clk              (clk),
        .reset            (reset),
    `ifdef PERF_ENABLE
        .tbuf_stalls    (shared_tbuf_stalls),
        .tbuf_cache_hits(shared_tbuf_cache_hits),
        .lmem_reads     (shared_lmem_reads),
    `endif
        .req_wid          (per_block_execute_if[0].data.header.wid),
        .req_valid        (req_valid_shared),
        .req_fire         (req_fire_shared),
        .req_is_sparse    (per_block_execute_if[0].data.op_args.tcu.is_sparse),
        .req_step_m       (per_block_execute_if[0].data.op_args.tcu.step_m),
        .req_step_n       (per_block_execute_if[0].data.op_args.tcu.step_n),
        .req_step_k       (per_block_execute_if[0].data.op_args.tcu.step_k),
        .req_fmt_s        (per_block_execute_if[0].data.op_args.tcu.fmt_s),
        .req_cd_nregs     (per_block_execute_if[0].data.op_args.tcu.cd_nregs),
        .req_desc_a       (per_block_execute_if[0].data.rs1_data[0]),
        .req_desc_b       (per_block_execute_if[0].data.rs2_data[0]),
        .req_a_is_smem    (per_block_execute_if[0].data.op_args.tcu.a_from_smem),
        .tcu_lmem_if      (tbuf_lmem_if),
        // Tile data outputs (broadcast to all cores)
        .tbuf_rs1_data    (shared_tbuf_rs1_data),
        .tbuf_rs2_data    (shared_tbuf_rs2_data),
    `ifdef TCU_SPARSE_ENABLE
        .tbuf_sp_meta     (shared_tbuf_sp_meta),
    `endif
        .tbuf_ready       (shared_tbuf_ready)
    );

    `ASSIGN_VX_MEM_BUS_IF (tcu_lmem_if, tbuf_lmem_if);

    // -------------------------------------------------------------------
    // Lockstep synchronization: WGMMA fires on ALL cores simultaneously
    // -------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0] core_wgmma_can_fire;
    wire [BLOCK_SIZE-1:0] core_wgmma_presenting;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wgmma_sync
        assign core_wgmma_presenting[bi] = per_block_execute_if[bi].valid
            && (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
    end
    wire wgmma_all_go = &core_wgmma_presenting && &core_wgmma_can_fire && shared_tbuf_ready;

    // -------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] shared_tbuf_stalls;
    wire [PERF_CTR_BITS-1:0] shared_tbuf_cache_hits;
    wire [PERF_CTR_BITS-1:0] shared_lmem_reads;
    assign tcu_perf.tbuf_stalls     = shared_tbuf_stalls;
    assign tcu_perf.tbuf_cache_hits = shared_tbuf_cache_hits;
    assign tcu_perf.lmem_reads      = shared_lmem_reads;

    // wgmma_instrs / wgmma_stalls: derived from per_block_execute_if.
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
            .tbuf_rs1_data (shared_tbuf_rs1_data),
            .tbuf_rs2_data (shared_tbuf_rs2_data),
        `ifdef TCU_SPARSE_ENABLE
            .tbuf_sp_meta  (shared_tbuf_sp_meta),
        `endif
            .tbuf_ready    (wgmma_all_go),
            .wgmma_can_fire(core_wgmma_can_fire[block_idx]),
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
