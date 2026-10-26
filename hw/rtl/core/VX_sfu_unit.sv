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

module VX_sfu_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if.slave     fpu_csr_if [`VX_CFG_NUM_FPU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_sched_unlock_if.master sched_unlock_if [`VX_CFG_NUM_SFU_BLOCKS],
`endif

    VX_sched_csr_if.slave   sched_csr_if [`VX_CFG_NUM_SFU_BLOCKS],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH],
    VX_warp_shared_if.master warp_shared_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_warp_ctl_if.master   warp_ctl_if [`VX_CFG_NUM_SFU_BLOCKS],

    input cta_bcast_t       cta_bcast
);
    localparam BLOCK_SIZE = `VX_CFG_NUM_SFU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_SFU_LANES;

    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (sfu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

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

`ifdef VX_CFG_EXT_F_ENABLE
    wire [BLOCK_SIZE-1:0][NW_WIDTH-1:0] fcsr_wid;
    wire [BLOCK_SIZE-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_rd_data;
    wire [BLOCK_SIZE-1:0][2:0] fcsr_wr_sel;
    wire [BLOCK_SIZE-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_wr_data;
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        VX_sfu_block #(
            .INSTANCE_ID ((BLOCK_SIZE > 1) ? `SFORMATF(("%s%0d", INSTANCE_ID, block_idx)) : INSTANCE_ID)
        ) sfu_block (
            .clk            (clk),
            .reset          (reset),
            .core_id        (core_id),
            .block_idx      (SFU_BLOCK_W'(block_idx)),
            .execute_if     (per_block_execute_if[block_idx]),
            .result_if      (per_block_result_if[block_idx]),
        `ifdef VX_CFG_EXT_F_ENABLE
            .fcsr_wid       (fcsr_wid[block_idx]),
            .fcsr_rd_data   (fcsr_rd_data[block_idx]),
            .fcsr_wr_sel    (fcsr_wr_sel[block_idx]),
            .fcsr_wr_data   (fcsr_wr_data[block_idx]),
        `endif
        `ifdef VX_CFG_EXT_DXA_ENABLE
            .dxa_req_bus_if (dxa_req_bus_if[block_idx]),
        `endif
        `ifdef VX_CFG_EXT_TEX_ENABLE
            .tex_bus_if     (tex_bus_if[block_idx]),
        `endif
        `ifdef VX_CFG_EXT_RTU_ENABLE
            .rtu_bus_if     (rtu_bus_if[block_idx]),
            .sched_unlock_if(sched_unlock_if[block_idx]),
        `endif
            .sched_csr_if   (sched_csr_if[block_idx]),
            .warp_shared_if (warp_shared_if[block_idx]),
            .warp_ctl_if    (warp_ctl_if[block_idx]),
            .cta_bcast      (cta_bcast)
        );
    end

`ifdef VX_CFG_EXT_F_ENABLE
    VX_csr_fpu #(
        .NUM_CSR_PORTS (BLOCK_SIZE),
        .NUM_FPU_PORTS (`VX_CFG_NUM_FPU_BLOCKS)
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

`ifdef VX_CFG_EXT_MUFU_ENABLE
    VX_mufu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-mufu", INSTANCE_ID))),
        .NUM_LANES   (NUM_LANES)
    ) mufu_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_MUFU]),
        .result_if  (pe_result_if[PE_IDX_MUFU])
    );
`endif

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
