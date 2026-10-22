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

`ifdef VX_CFG_EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

// The execution units of one issue slice: one ALU, LSU, SFU, FPU and TCU block
// each, and the FP CSRs of the slice's warps.
module VX_execute_slice import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
`ifdef VX_CFG_EXT_TCU_ENABLE
import VX_tcu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING SLICE_STR   = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire [ISSUE_ISW_W-1:0]   slice_id,

    VX_dispatch_if.slave    dispatch_if [NUM_EX_UNITS],
    VX_commit_if.master     commit_if [NUM_EX_UNITS],

    // ALU
    VX_branch_ctl_if.master branch_ctl_if,

    // LSU
`ifdef PERF_ENABLE
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rd_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_wr_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rsp_fire,
    output wire [PERF_CTR_BITS-1:0] lsu_coalescer_misses,
`endif
    output wire             lsu_sched_empty,
    output wire             lsu_store_empty,
    output wire             lsu_busy,
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.master    lmem_bus_if [`VX_CFG_NUM_LSU_LANES],
`endif
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_CHANNELS],

    // SFU
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
    VX_sched_unlock_if.master sched_unlock_if,
`endif

    // TCU
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    output wire             tcu_exec_valid,
    output wire             tcu_exec_ready,
    output tcu_execute_t    tcu_exec_data,
    input wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tcu_tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tcu_tbuf_rs2_data,
    input wire              tcu_tbuf_ready,
`endif

    // SFU warp control and CSRs
    VX_warp_shared_if.master warp_shared_if,
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_sched_csr_if.slave   sched_csr_if,
    input cta_bcast_t       cta_bcast
);
    localparam ALU_PARTIAL_BW = (`VX_CFG_NUM_ALU_LANES != `VX_CFG_SIMD_WIDTH);
`ifdef VX_CFG_EXT_F_ENABLE
    localparam FPU_PARTIAL_BW = (`VX_CFG_NUM_FPU_LANES != `VX_CFG_SIMD_WIDTH);
`endif

`ifdef SCOPE
    localparam SCOPE_COUNT = 1 + `VX_CFG_EXT_TCU_ENABLED;
`endif
    `SCOPE_IO_SWITCH (SCOPE_COUNT);

    // ALU

    VX_execute_if #(
        .data_t (alu_execute_t)
    ) alu_execute_if[1]();

    VX_result_if #(
        .data_t (alu_result_t)
    ) alu_result_if[1]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_ALU_LANES),
        .OUT_BUF    (ALU_PARTIAL_BW ? 3 : 0),
        .NUM_PORTS  (1)
    ) alu_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  (slice_id),
        .dispatch_if(dispatch_if[EX_ALU +: 1]),
        .execute_if (alu_execute_if)
    );

    VX_alu_block #(
        .INSTANCE_ID (`SFORMATF(("%s-alu", INSTANCE_ID))),
        .BLOCK_STR   (SLICE_STR),
        .NUM_LANES   (`VX_CFG_NUM_ALU_LANES),
        .PARTIAL_BW  (ALU_PARTIAL_BW)
    ) alu_block (
        .clk           (clk),
        .reset         (reset),
        .block_idx     (slice_id),
        .execute_if    (alu_execute_if[0]),
        .result_if     (alu_result_if[0]),
        .branch_ctl_if (branch_ctl_if)
    );

    VX_lane_gather #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_ALU_LANES),
        .OUT_BUF    (ALU_PARTIAL_BW ? 3 : 0),
        .NUM_PORTS  (1)
    ) alu_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (alu_result_if),
        .commit_if (commit_if[EX_ALU +: 1])
    );

    // LSU

    VX_execute_if #(
        .data_t (lsu_execute_t)
    ) lsu_execute_if[1]();

    VX_result_if #(
        .data_t (lsu_result_t)
    ) lsu_result_if[1]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_LSU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) lsu_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  (slice_id),
        .dispatch_if(dispatch_if[EX_LSU +: 1]),
        .execute_if (lsu_execute_if)
    );

`ifdef TCU_META_ENABLE
    // the TCU block's metadata loads go through this slice's LSU scheduler
    VX_lsu_sched_if lsu_tcu_mem_if();
`endif

    VX_lsu_block #(
        .INSTANCE_ID (`SFORMATF(("%s-lsu%s", INSTANCE_ID, SLICE_STR)))
    ) lsu_block (
        `SCOPE_IO_BIND  (0)
        .clk              (clk),
        .reset            (reset),
        .core_id          (core_id),
        .execute_if       (lsu_execute_if[0]),
        .result_if        (lsu_result_if[0]),
    `ifdef PERF_ENABLE
        .perf_rd_req_fire (lsu_perf_rd_req_fire),
        .perf_wr_req_fire (lsu_perf_wr_req_fire),
        .perf_rsp_fire    (lsu_perf_rsp_fire),
        .coalescer_misses (lsu_coalescer_misses),
    `endif
    `ifdef TCU_META_ENABLE
        .tcu_mem_if       (lsu_tcu_mem_if),
    `endif
        .sched_empty      (lsu_sched_empty),
        .store_empty      (lsu_store_empty),
        .busy             (lsu_busy),
    `ifdef VX_CFG_LMEM_ENABLE
        .lmem_bus_if      (lmem_bus_if),
    `endif
        .dcache_bus_if    (dcache_bus_if)
    );

    VX_lane_gather #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_LSU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) lsu_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (lsu_result_if),
        .commit_if (commit_if[EX_LSU +: 1])
    );

    // SFU

    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) sfu_execute_if[1]();

    VX_result_if #(
        .data_t (sfu_result_t)
    ) sfu_result_if[1]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_SFU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) sfu_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  (slice_id),
        .dispatch_if(dispatch_if[EX_SFU +: 1]),
        .execute_if (sfu_execute_if)
    );

`ifdef VX_CFG_EXT_F_ENABLE
    wire [NW_WIDTH-1:0] fcsr_wid;
    wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_rd_data;
    wire [2:0] fcsr_wr_sel;
    wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_wr_data;
`endif

    VX_sfu_block #(
        .INSTANCE_ID ((`VX_CFG_NUM_SFU_BLOCKS > 1) ? `SFORMATF(("%s-sfu%s", INSTANCE_ID, SLICE_STR)) : `SFORMATF(("%s-sfu", INSTANCE_ID)))
    ) sfu_block (
        .clk            (clk),
        .reset          (reset),
        .core_id        (core_id),
        .block_idx      (SFU_BLOCK_W'(slice_id)),
        .execute_if     (sfu_execute_if[0]),
        .result_if      (sfu_result_if[0]),
    `ifdef VX_CFG_EXT_F_ENABLE
        .fcsr_wid       (fcsr_wid),
        .fcsr_rd_data   (fcsr_rd_data),
        .fcsr_wr_sel    (fcsr_wr_sel),
        .fcsr_wr_data   (fcsr_wr_data),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if (dxa_req_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if     (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_bus_if     (rtu_bus_if),
        .sched_unlock_if(sched_unlock_if),
    `endif
        .sched_csr_if   (sched_csr_if),
        .warp_shared_if (warp_shared_if),
        .warp_ctl_if    (warp_ctl_if),
        .cta_bcast      (cta_bcast)
    );

    VX_lane_gather #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_SFU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) sfu_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (sfu_result_if),
        .commit_if (commit_if[EX_SFU +: 1])
    );

    // FPU

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if fpu_csr_if[1]();

    VX_execute_if #(
        .data_t (fpu_execute_t)
    ) fpu_execute_if[1]();

    VX_result_if #(
        .data_t (fpu_result_t)
    ) fpu_result_if[1]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_FPU_LANES),
        .OUT_BUF    (FPU_PARTIAL_BW ? 3 : 0),
        .NUM_PORTS  (1)
    ) fpu_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  (slice_id),
        .dispatch_if(dispatch_if[EX_FPU +: 1]),
        .execute_if (fpu_execute_if)
    );

    VX_fpu_block #(
        .INSTANCE_ID (`SFORMATF(("%s-fpu", INSTANCE_ID))),
        .NUM_LANES   (`VX_CFG_NUM_FPU_LANES),
        .PARTIAL_BW  (FPU_PARTIAL_BW)
    ) fpu_block (
        .clk        (clk),
        .reset      (reset),
        .block_idx  (slice_id),
        .execute_if (fpu_execute_if[0]),
        .result_if  (fpu_result_if[0]),
        .fpu_csr_if (fpu_csr_if[0])
    );

    VX_lane_gather #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_FPU_LANES),
        .OUT_BUF    (FPU_PARTIAL_BW ? 3 : 0),
        .NUM_PORTS  (1)
    ) fpu_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (fpu_result_if),
        .commit_if (commit_if[EX_FPU +: 1])
    );

    VX_csr_fpu #(
        .WID_STRIDE    (`VX_CFG_ISSUE_WIDTH),
        .NUM_CSR_PORTS (1),
        .NUM_FPU_PORTS (1)
    ) csr_fpu (
        .clk         (clk),
        .reset       (reset),
        .csr_wid     (fcsr_wid),
        .csr_rd_data (fcsr_rd_data),
        .csr_wr_sel  (fcsr_wr_sel),
        .csr_wr_data (fcsr_wr_data),
        .fpu_csr_if  (fpu_csr_if)
    );
`endif

    // TCU

`ifdef VX_CFG_EXT_TCU_ENABLE
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) tcu_execute_if[1]();

    VX_result_if #(
        .data_t (tcu_result_t)
    ) tcu_result_if[1]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_TCU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) tcu_dispatch (
        .clk        (clk),
        .reset      (reset),
        .port_base  (slice_id),
        .dispatch_if(dispatch_if[EX_TCU +: 1]),
        .execute_if (tcu_execute_if)
    );

    VX_tcu_block #(
        .INSTANCE_ID (`SFORMATF(("%s-tcu", INSTANCE_ID))),
        .BLOCK_STR   (SLICE_STR)
    ) tcu_block (
        `SCOPE_IO_BIND (1)
        .clk              (clk),
        .reset            (reset),
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .exec_valid       (tcu_exec_valid),
        .exec_ready       (tcu_exec_ready),
        .exec_data        (tcu_exec_data),
        .tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tbuf_ready       (tcu_tbuf_ready),
    `endif
    `ifdef TCU_META_ENABLE
        .tcu_mem_if       (lsu_tcu_mem_if),
    `endif
        .execute_if       (tcu_execute_if[0]),
        .result_if        (tcu_result_if[0])
    );

    VX_lane_gather #(
        .BLOCK_SIZE (1),
        .NUM_LANES  (`VX_CFG_NUM_TCU_LANES),
        .OUT_BUF    (3),
        .NUM_PORTS  (1)
    ) tcu_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (tcu_result_if),
        .commit_if (commit_if[EX_TCU +: 1])
    );
`endif

endmodule
