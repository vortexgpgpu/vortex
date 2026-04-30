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

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_inputs
        wire is_wgmma_b = (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        assign req_valid_arr    [bi] = per_block_execute_if[bi].valid && is_wgmma_b;
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
            .tbuf_ready    (tbuf_ready[block_idx]),
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
