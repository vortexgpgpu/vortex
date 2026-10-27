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

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // per-block ports of the core-wide WGMMA tile buffers (VX_tcu_shared)
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] exec_valid,
    output wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] exec_ready,
    output tcu_execute_t    exec_data [`VX_CFG_NUM_TCU_BLOCKS],
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data,
    input wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tbuf_ready,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory clients of the LSU blocks' schedulers
    VX_lsu_sched_if.master  tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
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
        .port_base  ('0),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (tcu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

`ifdef TCU_META_ENABLE
    VX_lsu_sched_if block_mem_if [BLOCK_SIZE]();

    if (`VX_CFG_NUM_LSU_BLOCKS == BLOCK_SIZE) begin : g_mem_per_block
        // each TCU block loads through the LSU block serving the same warps
        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_port
            assign tcu_mem_if[b].req_valid   = block_mem_if[b].req_valid;
            assign tcu_mem_if[b].req_data    = block_mem_if[b].req_data;
            assign block_mem_if[b].req_ready = tcu_mem_if[b].req_ready;
            assign block_mem_if[b].rsp_valid = tcu_mem_if[b].rsp_valid;
            assign block_mem_if[b].rsp_data  = tcu_mem_if[b].rsp_data;
            assign tcu_mem_if[b].rsp_ready   = block_mem_if[b].rsp_ready;
        end
    end else begin : g_mem_shared
        // the TCU blocks share LSU block 0's client, tagged by block
        localparam BLOCK_IDX_BITS = `CLOG2(BLOCK_SIZE);
        `STATIC_ASSERT (LSU_CLIENT_TAG_WIDTH >= BLOCK_IDX_BITS, ("LSU client tag cannot encode TCU block"))

        wire [BLOCK_SIZE-1:0] req_valids;
        wire [`UP(BLOCK_IDX_BITS)-1:0] req_idx;
        wire [BLOCK_SIZE-1:0] req_grant;
        wire req_valid;
        lsu_req_data_t req_datas [BLOCK_SIZE];
        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_req
            assign req_valids[b] = block_mem_if[b].req_valid;
            assign req_datas[b]  = block_mem_if[b].req_data;
            assign block_mem_if[b].req_ready = tcu_mem_if[0].req_ready && req_grant[b];
        end

        VX_generic_arbiter #(
            .NUM_REQS (BLOCK_SIZE),
            .TYPE     ("R")
        ) req_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (req_valids),
            .grant_index  (req_idx),
            .grant_onehot (req_grant),
            .grant_valid  (req_valid),
            .grant_ready  (tcu_mem_if[0].req_ready)
        );

        lsu_req_data_t req_data;
        always_comb begin
            req_data = req_datas[req_idx];
            req_data.tag[`UP(BLOCK_IDX_BITS)-1:0] = req_idx;
        end
        assign tcu_mem_if[0].req_valid = req_valid;
        assign tcu_mem_if[0].req_data  = req_data;

        wire [`UP(BLOCK_IDX_BITS)-1:0] rsp_idx = tcu_mem_if[0].rsp_data.tag[`UP(BLOCK_IDX_BITS)-1:0];
        wire [BLOCK_SIZE-1:0] rsp_readys;
        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_rsp
            assign block_mem_if[b].rsp_valid = tcu_mem_if[0].rsp_valid && (rsp_idx == `UP(BLOCK_IDX_BITS)'(b));
            assign block_mem_if[b].rsp_data  = tcu_mem_if[0].rsp_data;
            assign rsp_readys[b] = block_mem_if[b].rsp_ready;
        end
        assign tcu_mem_if[0].rsp_ready = rsp_readys[rsp_idx];

        for (genvar b = 1; b < `VX_CFG_NUM_LSU_BLOCKS; ++b) begin : g_tieoff
            assign tcu_mem_if[b].req_valid = 1'b0;
            assign tcu_mem_if[b].req_data  = '0;
            assign tcu_mem_if[b].rsp_ready = 1'b1;
            `UNUSED_VAR (tcu_mem_if[b].req_ready)
            `UNUSED_VAR (tcu_mem_if[b].rsp_valid)
            `UNUSED_VAR (tcu_mem_if[b].rsp_data)
        end
    end
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks

        VX_tcu_block #(
            .INSTANCE_ID (INSTANCE_ID),
            .BLOCK_STR   (`SFORMATF(("%0d", block_idx)))
        ) tcu_block (
            `SCOPE_IO_BIND (block_idx)
            .clk              (clk),
            .reset            (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .exec_valid       (exec_valid[block_idx]),
            .exec_ready       (exec_ready[block_idx]),
            .exec_data        (exec_data[block_idx]),
            .tbuf_rs1_data    (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data    (tbuf_rs2_data[block_idx]),
            .tbuf_ready       (tbuf_ready[block_idx]),
        `endif
        `ifdef TCU_META_ENABLE
            .tcu_mem_if       (block_mem_if[block_idx]),
        `endif
            .execute_if       (per_block_execute_if[block_idx]),
            .result_if        (per_block_result_if[block_idx])
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
