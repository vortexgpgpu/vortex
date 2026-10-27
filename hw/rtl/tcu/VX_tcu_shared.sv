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

// TCU state shared by every TCU block: the WGMMA tile buffers (a warpgroup's
// B operand is common to its blocks).
module VX_tcu_shared import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS
) (
`ifdef PERF_ENABLE
    output tcu_perf_t       tcu_perf,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    input wire [BLOCK_SIZE-1:0] exec_valid,
    input wire [BLOCK_SIZE-1:0] exec_ready,
    input tcu_execute_t     exec_data [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data,
    output wire [BLOCK_SIZE-1:0] tbuf_ready_eff,
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master    tcu_lmem_if,
`endif

    input wire              clk,
    input wire              reset
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifndef VX_CFG_TCU_WGMMA_ENABLE
    `UNUSED_PARAM (BLOCK_SIZE)
    `UNUSED_VAR ({clk, reset})
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    VX_tcu_wgmma #(
        .INSTANCE_ID (`SFORMATF(("%s-wgmma", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) wgmma (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
        .exec_valid     (exec_valid),
        .exec_ready     (exec_ready),
        .exec_data      (exec_data),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready_eff (tbuf_ready_eff)
    );
`else
`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = '0;
    assign tcu_perf.tbuf_cache_hits = '0;
    assign tcu_perf.lmem_reads      = '0;
    assign tcu_perf.wgmma_instrs    = '0;
    assign tcu_perf.wgmma_stalls    = '0;
`endif
`endif

endmodule
