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
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
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

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // Tensor-job barrier operations and stage loads
    VX_tcu_bar_if.master    tcu_bar_if,
    VX_dxa_req_bus_if.master tcu_dxa_req_bus_if,
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
    `UNUSED_PARAM (CORE_ID)

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
    // TMEM transfers and tensor-job posts go to the job unit; job MMA steps
    // take a block's tcu_core ahead of warp MMAs.
    wire [BLOCK_SIZE-1:0] tj_op_valid;
    wire [BLOCK_SIZE-1:0] tj_op_ready;
    tcu_execute_t         tj_op_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] tj_rsp_valid;
    tcu_result_t          tj_rsp_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] tj_rsp_ready;
    wire                  job_valid;
    wire                  job_fire;
    tcu_execute_t         job_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] job_core_ready;
    tcu_tbuf_req_t [BLOCK_SIZE-1:0] job_tbuf_req;
    wire [BLOCK_SIZE-1:0] jres_valid;
    wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] jres_data;
`endif

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_split
        wire [INST_OP_BITS-1:0] op_type = per_block_execute_if[bi].data.op_type;
        `UNUSED_VAR (op_type)
    `ifdef TCU_META_ENABLE
        wire is_tcu_ld = (op_type == INST_OP_BITS'(INST_TCU_LD));
    `else
        wire is_tcu_ld = 1'b0;
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        wire is_tj_op = (op_type == INST_OP_BITS'(INST_TCU_TMEM_LD))
                     || (op_type == INST_OP_BITS'(INST_TCU_TMEM_ST))
                     || (op_type == INST_OP_BITS'(INST_TCU_TJ_DESC))
                     || (op_type == INST_OP_BITS'(INST_TCU_TJ_POST));
        assign tj_op_valid[bi] = per_block_execute_if[bi].valid && is_tj_op;
        assign tj_op_data[bi]  = per_block_execute_if[bi].data;
        wire warp_core_valid = per_block_execute_if[bi].valid && !is_tcu_ld && !is_tj_op;
        assign core_execute_if[bi].valid = job_valid ? job_fire : warp_core_valid;
        assign core_execute_if[bi].data  = job_valid ? job_data[bi] : per_block_execute_if[bi].data;
        assign job_core_ready[bi]        = core_execute_if[bi].ready;
        wire warp_core_ready = core_execute_if[bi].ready && !job_valid;
        `RUNTIME_ASSERT (~(warp_core_valid && (op_type == INST_OP_BITS'(INST_TCU_WGMMA)
                        `ifdef VX_CFG_TCU_SPARSE_ENABLE
                            || op_type == INST_OP_BITS'(INST_TCU_WGMMA_SP)
                        `endif
                        )), ("%t: %s: WGMMA is not supported with tensor memory", $time, INSTANCE_ID))
    `else
        wire is_tj_op = 1'b0;
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid && !is_tcu_ld;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;
        wire warp_core_ready = core_execute_if[bi].ready;
    `endif
    `ifdef TCU_META_ENABLE
        assign agu_ld_valid[bi] = per_block_execute_if[bi].valid && is_tcu_ld;
        assign agu_ld_data[bi]  = per_block_execute_if[bi].data;
        wire agu_ready = agu_ld_ready[bi];
    `else
        wire agu_ready = 1'b0;
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        wire tj_ready = tj_op_ready[bi];
    `else
        wire tj_ready = 1'b0;
    `endif
        assign per_block_execute_if[bi].ready = is_tcu_ld ? agu_ready
                                              : is_tj_op  ? tj_ready
                                              : warp_core_ready;
    end

    // -----------------------------------------------------------------------
    // Result merge per block: AGU, then the job unit, then tcu_core. Job MMA
    // results leave tcu_core in issue order and are written to TMEM.
    // -----------------------------------------------------------------------
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_merge
    `ifdef TCU_META_ENABLE
        wire a_valid = agu_result_valid[bi];
        tcu_result_t a_data;
        assign a_data = agu_result_data[bi];
        assign agu_result_ready[bi] = per_block_result_if[bi].ready;
    `else
        wire a_valid = 1'b0;
        tcu_result_t a_data;
        assign a_data = 'x;
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        wire t_valid = tj_rsp_valid[bi];
        tcu_result_t t_data;
        assign t_data = tj_rsp_data[bi];
        assign tj_rsp_ready[bi] = per_block_result_if[bi].ready && !a_valid;

        // Which tcu_core results belong to jobs, in issue order.
        wire is_job_head, order_empty, order_full;
        wire core_in_fire  = core_execute_if[bi].valid && core_execute_if[bi].ready;
        wire core_out_fire = core_result_if[bi].valid && core_result_if[bi].ready;
        VX_fifo_queue #(
            .DATAW  (1),
            .DEPTH  (32),
            .LUTRAM (1)
        ) order_q (
            .clk     (clk),
            .reset   (reset),
            .push    (core_in_fire),
            .pop     (core_out_fire),
            .data_in (job_valid),
            .data_out(is_job_head),
            .empty   (order_empty),
            `UNUSED_PIN (alm_empty),
            .full    (order_full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
        `UNUSED_VAR (order_empty)
        `RUNTIME_ASSERT (~(core_in_fire && order_full), ("%t: %s: tcu order queue overflow", $time, INSTANCE_ID))
        wire job_res = core_result_if[bi].valid && is_job_head;
        assign jres_valid[bi] = job_res;
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_jres
            assign jres_data[bi][t] = core_result_if[bi].data.data[t][31:0];
        end
        wire c_valid = core_result_if[bi].valid && !is_job_head;
        assign core_result_if[bi].ready = is_job_head
            || (per_block_result_if[bi].ready && !a_valid && !t_valid);
    `else
        wire t_valid = 1'b0;
        tcu_result_t t_data;
        assign t_data = 'x;
        wire c_valid = core_result_if[bi].valid;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !a_valid;
    `endif
        assign per_block_result_if[bi].valid = a_valid || t_valid || c_valid;
        assign per_block_result_if[bi].data  = a_valid ? a_data
                                             : t_valid ? t_data
                                             : core_result_if[bi].data;
    end

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
        .job_req        (job_tbuf_req),
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
    VX_tcu_tjob #(
        .INSTANCE_ID (`SFORMATF(("%s-tjob", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) tjob (
        .clk            (clk),
        .reset          (reset),
        .op_valid       (tj_op_valid),
        .op_data        (tj_op_data),
        .op_ready       (tj_op_ready),
        .rsp_valid      (tj_rsp_valid),
        .rsp_data       (tj_rsp_data),
        .rsp_ready      (tj_rsp_ready),
        .job_valid      (job_valid),
        .job_data       (job_data),
        .core_ready     (job_core_ready),
        .job_fire       (job_fire),
        .tbuf_req       (job_tbuf_req),
        .jres_valid     (jres_valid),
        .jres_data      (jres_data),
        .tcu_bar_if     (tcu_bar_if),
        .dxa_req_bus_if (tcu_dxa_req_bus_if)
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
