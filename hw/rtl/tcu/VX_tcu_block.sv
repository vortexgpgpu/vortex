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

// One TCU block: the MMA datapath, with TCU_LD handed to the core-wide AGU and
// WGMMA operands supplied by the core-wide tile buffers (VX_tcu_shared).
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
    // TCU_LD to and from the AGU
    output wire             agu_ld_valid,
    output tcu_execute_t    agu_ld_data,
    input wire              agu_ld_ready,
    input wire              agu_result_valid,
    input tcu_result_t      agu_result_data,
    output wire             agu_result_ready,
    input wire              meta_wr_en,
    input wire [NW_WIDTH-1:0] meta_wr_wid,
    input wire [4:0]        meta_wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] meta_wr_data,
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
    // TCU_LD goes to the AGU, every MMA op to the core
    wire is_tcu_ld = (execute_if.data.op_type == INST_TCU_LD);

    assign agu_ld_valid          = execute_if.valid && is_tcu_ld;
    assign agu_ld_data           = execute_if.data;
    assign core_execute_if.valid = execute_if.valid && !is_tcu_ld;
    assign core_execute_if.data  = execute_if.data;
    assign execute_if.ready      = is_tcu_ld ? agu_ld_ready : core_execute_if.ready;

    // AGU wins same-cycle conflicts; the core stalls and retries next cycle.
    assign result_if.valid       = agu_result_valid || core_result_if.valid;
    assign result_if.data        = agu_result_valid ? agu_result_data : core_result_if.data;
    assign agu_result_ready      = result_if.ready;
    assign core_result_if.ready  = result_if.ready && !agu_result_valid;
`else
    assign core_execute_if.valid = execute_if.valid;
    assign core_execute_if.data  = execute_if.data;
    assign execute_if.ready      = core_execute_if.ready;

    assign result_if.valid       = core_result_if.valid;
    assign result_if.data        = core_result_if.data;
    assign core_result_if.ready  = result_if.ready;
`endif

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
        .ext_meta_wr_data (meta_wr_data),
    `endif
        .execute_if (core_execute_if),
        .result_if  (core_result_if)
    );

endmodule
