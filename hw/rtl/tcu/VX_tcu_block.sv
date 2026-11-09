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

// One TCU block: the MMA datapath and the TCU_LD unit that fills its metadata
// RAM, with WGMMA operands supplied by the core-wide tile buffers
// (VX_tcu_shared).
module VX_tcu_block import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING BLOCK_STR   = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // MMA execute handshake observed by the WGMMA tile buffers
    output wire             exec_valid,
    output wire             exec_ready,
    output tcu_execute_t    exec_data,
    input wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data,
    input wire              tbuf_ready,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory client of an LSU scheduler
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    VX_tcu_job_if.block     job_if,
`endif

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if
);
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_BLOCKS == `VX_CFG_ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_LANES == `VX_CFG_NUM_THREADS, ("must be full warp execution"));

    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) core_execute_if();

    VX_result_if #(
        .data_t (tcu_result_t)
    ) core_result_if();

`ifdef TCU_META_ENABLE
    wire                    agu_ld_valid, agu_ld_ready;
    wire                    agu_result_valid, agu_result_ready;
    tcu_result_t            agu_result_data;
    wire                    meta_wr_en;
    wire [NW_WIDTH-1:0]     meta_wr_wid;
    wire [4:0]              meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0] meta_wr_mask;
    wire [TCU_BLOCK_CAP-1:0][31:0] meta_wr_data;

    VX_tcu_agu #(
        .INSTANCE_ID (`SFORMATF(("%s-agu%s", INSTANCE_ID, BLOCK_STR)))
    ) agu (
        .clk          (clk),
        .reset        (reset),
        .ld_valid     (agu_ld_valid),
        .ld_data      (execute_if.data),
        .ld_ready     (agu_ld_ready),
        .client_if    (tcu_mem_if),
        .meta_wr_en   (meta_wr_en),
        .meta_wr_wid  (meta_wr_wid),
        .meta_wr_idx  (meta_wr_idx),
        .meta_wr_mask (meta_wr_mask),
        .meta_wr_data (meta_wr_data),
        .result_valid (agu_result_valid),
        .result_data  (agu_result_data),
        .result_ready (agu_result_ready)
    );

`endif

    // TCU_LD goes to the AGU, TMEM and tc ops to the tensor memory, every MMA
    // op to the core.
    wire [INST_OP_BITS-1:0] op_type = execute_if.data.op_type;
    `UNUSED_VAR (op_type)
`ifdef TCU_META_ENABLE
    wire is_tcu_ld = (op_type == INST_TCU_LD);
`else
    wire is_tcu_ld = 1'b0;
`endif
`ifdef VX_CFG_TCU_TMEM_ENABLE
    wire is_tj_op = (op_type == INST_OP_BITS'(INST_TCU_TMEM_LD))
                 || (op_type == INST_OP_BITS'(INST_TCU_TMEM_ST))
                 || (op_type == INST_OP_BITS'(INST_TCU_TC_CP))
                 || (op_type == INST_OP_BITS'(INST_TCU_TMEM_ALLOC))
                 || (op_type == INST_OP_BITS'(INST_TCU_TMEM_DEALLOC))
                 || (op_type == INST_OP_BITS'(INST_TCU_TC_IDESC))
                 || (op_type == INST_OP_BITS'(INST_TCU_TC_MMA))
                 || (op_type == INST_OP_BITS'(INST_TCU_TC_COMMIT));
`else
    wire is_tj_op = 1'b0;
`endif
    wire warp_core_valid = execute_if.valid && !is_tcu_ld && !is_tj_op;

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // Tensor-op MMA steps take the core ahead of warp MMAs.
    assign job_if.op_valid       = execute_if.valid && is_tj_op;
    assign job_if.op_data        = execute_if.data;
    assign core_execute_if.valid = job_if.job_valid ? job_if.job_fire : warp_core_valid;
    assign core_execute_if.data  = job_if.job_valid ? job_if.job_data : execute_if.data;
    assign job_if.core_ready     = core_execute_if.ready;
    wire warp_core_ready = core_execute_if.ready && !job_if.job_valid;
    wire tj_ready        = job_if.op_ready;
    `RUNTIME_ASSERT (~(warp_core_valid && (op_type == INST_OP_BITS'(INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                        || op_type == INST_OP_BITS'(INST_TCU_WGMMA_SP)
                    `endif
                    )), ("%t: %s: WGMMA is not supported with tensor memory", $time, INSTANCE_ID))
`else
    assign core_execute_if.valid = warp_core_valid;
    assign core_execute_if.data  = execute_if.data;
    wire warp_core_ready = core_execute_if.ready;
    wire tj_ready        = 1'b0;
`endif
`ifdef TCU_META_ENABLE
    assign agu_ld_valid = execute_if.valid && is_tcu_ld;
    wire agu_ready = agu_ld_ready;
`else
    wire agu_ready = 1'b0;
`endif
    assign execute_if.ready = is_tcu_ld ? agu_ready
                            : is_tj_op  ? tj_ready
                            : warp_core_ready;

    // Results: the AGU, then the tensor memory, then the core. A tensor-op
    // MMA step's result goes back to the tensor memory, in issue order.
`ifdef TCU_META_ENABLE
    wire a_valid = agu_result_valid;
    tcu_result_t a_data;
    assign a_data = agu_result_data;
    assign agu_result_ready = result_if.ready;
`else
    wire a_valid = 1'b0;
    tcu_result_t a_data;
    assign a_data = 'x;
`endif
`ifdef VX_CFG_TCU_TMEM_ENABLE
    wire t_valid = job_if.rsp_valid;
    tcu_result_t t_data;
    assign t_data = job_if.rsp_data;
    assign job_if.rsp_ready = result_if.ready && !a_valid;

    wire is_job_head, order_empty, order_full;
    wire core_in_fire  = core_execute_if.valid && core_execute_if.ready;
    wire core_out_fire = core_result_if.valid && core_result_if.ready;
    VX_fifo_queue #(
        .DATAW  (1),
        .DEPTH  (32),
        .LUTRAM (1)
    ) order_q (
        .clk     (clk),
        .reset   (reset),
        .push    (core_in_fire),
        .pop     (core_out_fire),
        .data_in (job_if.job_valid),
        .data_out(is_job_head),
        .empty   (order_empty),
        `UNUSED_PIN (alm_empty),
        .full    (order_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    `UNUSED_VAR (order_empty)
    `RUNTIME_ASSERT (~(core_in_fire && order_full), ("%t: %s: tcu order queue overflow", $time, INSTANCE_ID))
    assign job_if.jres_valid = core_result_if.valid && is_job_head;
    for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_jres
        assign job_if.jres_data[t] = core_result_if.data.data[t][31:0];
    end
    wire c_valid = core_result_if.valid && !is_job_head;
    assign core_result_if.ready = is_job_head || (result_if.ready && !a_valid && !t_valid);
`else
    wire t_valid = 1'b0;
    tcu_result_t t_data;
    assign t_data = 'x;
    wire c_valid = core_result_if.valid;
    assign core_result_if.ready = result_if.ready && !a_valid;
`endif
    assign result_if.valid = a_valid || t_valid || c_valid;
    assign result_if.data  = a_valid ? a_data
                           : t_valid ? t_data
                           : core_result_if.data;

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    assign exec_valid = core_execute_if.valid;
    assign exec_ready = core_execute_if.ready;
    assign exec_data  = core_execute_if.data;
`endif

    VX_tcu_core #(
        .INSTANCE_ID (`SFORMATF(("%s-fused%s", INSTANCE_ID, BLOCK_STR)))
    ) tcu_core (
    `ifdef SCOPE
        .scope_reset   (scope_reset),
        .scope_bus_in  (scope_bus_in),
        .scope_bus_out (scope_bus_out),
    `endif
        .clk        (clk),
        .reset      (reset),
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tbuf_rs1_data (tbuf_rs1_data),
        .tbuf_rs2_data (tbuf_rs2_data),
        .tbuf_ready    (tbuf_ready),
    `endif
    `ifdef TCU_META_ENABLE
        .ext_meta_wr_en   (meta_wr_en),
        .ext_meta_wr_wid  (meta_wr_wid),
        .ext_meta_wr_idx  (meta_wr_idx),
        .ext_meta_wr_mask (meta_wr_mask),
        .ext_meta_wr_data (meta_wr_data),
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        .job_meta         (job_if.job_meta),
        .job_meta_hi      (job_if.job_meta_hi),
    `endif
    `endif
        .execute_if (core_execute_if),
        .result_if  (core_result_if)
    );

endmodule
