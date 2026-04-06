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
    // WGMMA tile buffers + LMEM port arbitration
    // -----------------------------------------------------------------------

`ifdef TCU_WGMMA_ENABLE
    localparam BANK_ADDR_WIDTH = `LMEM_LOG_SIZE - $clog2(`XLEN / 8) - $clog2(`LMEM_NUM_BANKS);

    // Per-block tile buffer outputs
    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data  [BLOCK_SIZE];
    wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data [BLOCK_SIZE];
`ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta  [BLOCK_SIZE];
`endif
    wire                                tbuf_ready     [BLOCK_SIZE];

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0]     tbuf_fetch_stalls_b [BLOCK_SIZE];
    wire [PERF_CTR_BITS-1:0]     lmem_reads_b        [BLOCK_SIZE];
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`LMEM_NUM_BANKS * (`XLEN / 8)),
        .TAG_WIDTH (LMEM_DMA_TAG_W),
        .ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) per_block_lmem_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tile_bufs
        if (BLOCK_SIZE == 1) begin : g_rsp_direct
            assign per_block_lmem_if[block_idx].rsp_valid = tcu_lmem_if.rsp_valid;
        end else begin : g_rsp_tagged
            // Route responses only to the block that made the request (via tag)
            assign per_block_lmem_if[block_idx].rsp_valid = tcu_lmem_if.rsp_valid
                && (tcu_lmem_if.rsp_data.tag[$clog2(BLOCK_SIZE)-1:0] == $clog2(BLOCK_SIZE)'(block_idx));
        end
        assign per_block_lmem_if[block_idx].rsp_data  = tcu_lmem_if.rsp_data;

        wire is_wgmma_b = (per_block_execute_if[block_idx].data.op_type == INST_TCU_WGMMA);

        wire req_valid_b = per_block_execute_if[block_idx].valid && is_wgmma_b;
        wire req_fire_b  = per_block_execute_if[block_idx].valid
                        && per_block_execute_if[block_idx].ready
                        && is_wgmma_b;

        VX_tcu_tbuf #(
            .INSTANCE_ID    (`SFORMATF(("%s-tbuf%0d", INSTANCE_ID, block_idx))),
            .NUM_BANKS      (`LMEM_NUM_BANKS),
            .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
        ) tile_buf (
            .clk              (clk),
            .reset            (reset),
        `ifdef PERF_ENABLE
            .tbuf_fetch_stalls(tbuf_fetch_stalls_b[block_idx]),
            .lmem_reads       (lmem_reads_b[block_idx]),
        `endif
            .req_valid        (req_valid_b),
            .req_fire         (req_fire_b),
            .req_is_sparse    (per_block_execute_if[block_idx].data.op_args.tcu.is_sparse),
            .req_step_m       (per_block_execute_if[block_idx].data.op_args.tcu.step_m),
            .req_step_n       (per_block_execute_if[block_idx].data.op_args.tcu.step_n),
            .req_step_k       (per_block_execute_if[block_idx].data.op_args.tcu.step_k),
            .req_fmt_s        (per_block_execute_if[block_idx].data.op_args.tcu.fmt_s),
            .req_cd_nregs     (per_block_execute_if[block_idx].data.op_args.tcu.cd_nregs),
            .req_desc_a       (per_block_execute_if[block_idx].data.rs1_data[0]),
            .req_desc_b       (per_block_execute_if[block_idx].data.rs2_data[0]),
            .tcu_lmem_if      (per_block_lmem_if[block_idx]),
            // Tile data outputs
            .tbuf_rs1_data    (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data    (tbuf_rs2_data[block_idx]),
        `ifdef TCU_SPARSE_ENABLE
            .tbuf_sp_meta     (tbuf_sp_meta[block_idx]),
        `endif
            .tbuf_ready       (tbuf_ready[block_idx])
        );
    end

    // -------------------------------------------------------------------
    // LMEM port arbitration
    // -------------------------------------------------------------------
    //   For BLOCK_SIZE==1 (typical), this is a direct wire-through.
    //   For BLOCK_SIZE>1, a simple priority arbiter grants the port
    //   to one tile buffer at a time.

    if (BLOCK_SIZE == 1) begin : g_lmem_direct
        assign tcu_lmem_if.req_valid       = per_block_lmem_if[0].req_valid;
        assign per_block_lmem_if[0].req_ready = tcu_lmem_if.req_ready;
        assign tcu_lmem_if.req_data.rw     = 1'b0;
        assign tcu_lmem_if.req_data.addr   = per_block_lmem_if[0].req_data.addr;
        assign tcu_lmem_if.req_data.data   = '0;
        assign tcu_lmem_if.req_data.byteen = '0;
        assign tcu_lmem_if.req_data.flags  = '0;
        assign tcu_lmem_if.req_data.tag    = '0;
        assign tcu_lmem_if.rsp_ready       = 1'b1;
        `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)
    end else begin : g_lmem_arb
        // Extract interface signals into plain arrays for variable indexing
        wire [BLOCK_SIZE-1:0] blk_req_valid;
        wire [BANK_ADDR_WIDTH-1:0] blk_req_addr [BLOCK_SIZE];
        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_blk_sigs
            assign blk_req_valid[b] = per_block_lmem_if[b].req_valid;
            assign blk_req_addr[b]  = per_block_lmem_if[b].req_data.addr;
        end

        // Priority arbiter: lowest block index wins
        logic [$clog2(BLOCK_SIZE)-1:0] grant_idx;
        logic                          grant_valid;
        logic [BANK_ADDR_WIDTH-1:0]    grant_addr;

        always_comb begin
            grant_idx   = '0;
            grant_valid = 1'b0;
            grant_addr  = '0;
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (blk_req_valid[b] && !grant_valid) begin
                    grant_idx   = $clog2(BLOCK_SIZE)'(b);
                    grant_valid = 1'b1;
                    grant_addr  = blk_req_addr[b];
                end
            end
        end

        assign tcu_lmem_if.req_valid       = grant_valid;
        assign tcu_lmem_if.req_data.rw     = 1'b0;
        assign tcu_lmem_if.req_data.addr   = grant_addr;
        assign tcu_lmem_if.req_data.data   = '0;
        assign tcu_lmem_if.req_data.byteen = '0;
        assign tcu_lmem_if.req_data.flags  = '0;
        assign tcu_lmem_if.req_data.tag    = LMEM_DMA_TAG_W'(grant_idx);
        assign tcu_lmem_if.rsp_ready       = 1'b1;

        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_rd_ready
            assign per_block_lmem_if[b].req_ready = tcu_lmem_if.req_ready
                                                 && grant_valid
                                                 && (grant_idx == $clog2(BLOCK_SIZE)'(b));
        end
    end

    // -------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------

`ifdef PERF_ENABLE
    logic [PERF_CTR_BITS-1:0] tbuf_fetch_stalls_sum;
    logic [PERF_CTR_BITS-1:0] lmem_reads_sum;
    always_comb begin
        tbuf_fetch_stalls_sum = '0;
        lmem_reads_sum        = '0;
        for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
            tbuf_fetch_stalls_sum += tbuf_fetch_stalls_b[bi];
            lmem_reads_sum        += lmem_reads_b[bi];
        end
    end
    assign tcu_perf.tbuf_fetch_stalls = tbuf_fetch_stalls_sum;
    assign tcu_perf.lmem_reads        = lmem_reads_sum;

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
    assign tcu_perf.tbuf_fetch_stalls = '0;
    assign tcu_perf.lmem_reads        = '0;
    assign tcu_perf.wgmma_instrs      = '0;
    assign tcu_perf.wgmma_stalls      = '0;
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
            .tbuf_ready (tbuf_ready[block_idx]),
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
