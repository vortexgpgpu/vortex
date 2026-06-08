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

module VX_execute import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
    input pipeline_perf_t   pipeline_perf,
`ifdef VX_CFG_EXT_TCU_ENABLE
    output tcu_perf_t       tcu_perf,
`endif
`endif

    // Per-block LSU client connections to VX_lsu_scheduler (at VX_core).
    // The dcache port (lsu_mem_if) lives at VX_core; execute exposes
    // only the LSU client interfaces.
    VX_lsu_sched_if.master lsu_client_if [`VX_CFG_NUM_LSU_BLOCKS],

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // TCU AGU memory client (single warp-level AGU shared across blocks).
    // VX_core wires this to client 1 of block 0's lsu_scheduler.
    VX_lsu_sched_if.master tcu_mem_if,
`endif

    // dispatch interface
    VX_dispatch_if.slave    dispatch_if [NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH],

    // commit interface
    VX_commit_if.master     commit_if [NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH],

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_txbar_bus_if.slave  dxa_txbar_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_om_bus_if.master     om_bus_if,
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if.slave  raster_bus_if,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
    VX_async_trap_if.master async_trap_if,
`endif

    // scheduler interfaces
    VX_sched_csr_if.slave   sched_csr_if,
    VX_branch_ctl_if.master branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS],
    VX_warp_ctl_if.master   warp_ctl_if,

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // TCU tile-buffer local-memory read port
    VX_mem_bus_if.master    tcu_lmem_if,
`endif

    // DCR-CSR interface
    VX_dcr_csr_if           dcr_csr_if
);

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if fpu_csr_if[`VX_CFG_NUM_FPU_BLOCKS]();
`endif

    VX_alu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-alu", INSTANCE_ID)))
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_ALU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_ALU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .branch_ctl_if  (branch_ctl_if)
    );

    `SCOPE_IO_SWITCH (1);

    VX_lsu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-lsu", INSTANCE_ID))),
        .CORE_ID     (CORE_ID)
    ) lsu_unit (
        `SCOPE_IO_BIND  (0)
        .clk             (clk),
        .reset           (reset),
        .dispatch_if         (dispatch_if[EX_LSU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if           (commit_if[EX_LSU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .per_block_client_if (lsu_client_if)
    );

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-fpu", INSTANCE_ID)))
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_FPU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_FPU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .fpu_csr_if     (fpu_csr_if)
    );
`endif

`ifdef VX_CFG_EXT_TCU_ENABLE
    VX_tcu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tcu", INSTANCE_ID)))
    ) tcu_unit (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_lmem_if    (tcu_lmem_if),
    `endif
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        .tcu_mem_if        (tcu_mem_if),
    `endif
        .dispatch_if    (dispatch_if[EX_TCU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_TCU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH])
    );
`endif

    VX_sfu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-sfu", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) sfu_unit (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
        .pipeline_perf  (pipeline_perf),
    `endif
        .dispatch_if    (dispatch_if[EX_SFU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_SFU * `VX_CFG_ISSUE_WIDTH +: `VX_CFG_ISSUE_WIDTH]),
    `ifdef VX_CFG_EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_req_bus_if (dxa_req_bus_if),
        .dxa_txbar_bus_if(dxa_txbar_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .tex_bus_if     (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        .om_bus_if      (om_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_bus_if  (raster_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .rtu_bus_if     (rtu_bus_if),
        .async_trap_if  (async_trap_if),
    `endif
        .sched_csr_if   (sched_csr_if),
        .warp_ctl_if    (warp_ctl_if),
        .dcr_csr_if     (dcr_csr_if)
    );

endmodule
