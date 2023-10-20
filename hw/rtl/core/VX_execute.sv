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
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk, 
    input wire              reset,    

    input base_dcrs_t       base_dcrs,

    // Dcache interface
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    // commit interface
    VX_commit_csr_if.slave  commit_csr_if,

    // fetch interface
    VX_sched_csr_if.slave   sched_csr_if,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
    VX_pipeline_perf_if.slave pipeline_perf_if,
`endif

`ifdef EXT_F_ENABLE
    VX_dispatch_if.slave    fpu_dispatch_if [`ISSUE_WIDTH],
    VX_commit_if.master     fpu_commit_if [`ISSUE_WIDTH],
`endif
  
    VX_dispatch_if.slave    alu_dispatch_if [`ISSUE_WIDTH],
    VX_commit_if.master     alu_commit_if [`ISSUE_WIDTH],
    VX_branch_ctl_if.master branch_ctl_if [`NUM_ALU_BLOCKS],
    
    VX_dispatch_if.slave    lsu_dispatch_if [`ISSUE_WIDTH],  
    VX_commit_if.master     lsu_commit_if [`ISSUE_WIDTH],
    
    VX_dispatch_if.slave    sfu_dispatch_if [`ISSUE_WIDTH], 
    VX_commit_if.master     sfu_commit_if [`ISSUE_WIDTH],
    VX_warp_ctl_if.master   warp_ctl_if,

    // simulation helper signals
    output wire             sim_ebreak
);

`ifdef EXT_F_ENABLE
    VX_fpu_to_csr_if fpu_to_csr_if[`NUM_FPU_BLOCKS]();
`endif

    `RESET_RELAY (alu_reset, reset);
    `RESET_RELAY (lsu_reset, reset);
    `RESET_RELAY (sfu_reset, reset);
    
    VX_alu_unit #(
        .CORE_ID (CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (alu_reset),
        .dispatch_if    (alu_dispatch_if),
        .branch_ctl_if  (branch_ctl_if),
        .commit_if      (alu_commit_if)
    );

    `SCOPE_IO_SWITCH (1)

    VX_lsu_unit #(
        .CORE_ID (CORE_ID)
    ) lsu_unit (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (lsu_reset),
        .cache_bus_if   (dcache_bus_if),
        .dispatch_if    (lsu_dispatch_if),
        .commit_if      (lsu_commit_if)
    );

`ifdef EXT_F_ENABLE
    `RESET_RELAY (fpu_reset, reset);

    VX_fpu_unit #(
        .CORE_ID (CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (fpu_reset),    
        .dispatch_if    (fpu_dispatch_if), 
        .fpu_to_csr_if  (fpu_to_csr_if),
        .commit_if      (fpu_commit_if)
    );
`endif

    VX_sfu_unit #(
        .CORE_ID (CORE_ID)
    ) sfu_unit (
        .clk            (clk),
        .reset          (sfu_reset),

    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_if),
        .pipeline_perf_if (pipeline_perf_if),
    `endif

        .base_dcrs      (base_dcrs),            

        .dispatch_if    (sfu_dispatch_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_to_csr_if  (fpu_to_csr_if),
    `endif
    
        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),
        .warp_ctl_if    (warp_ctl_if),
        .commit_if      (sfu_commit_if)
    );

    // simulation helper signal to get RISC-V tests Pass/Fail status
    assign sim_ebreak = alu_dispatch_if[0].valid && alu_dispatch_if[0].ready 
                     && alu_dispatch_if[0].data.wis == 0
                     && `INST_ALU_IS_BR(alu_dispatch_if[0].data.op_mod)
                     && (`INST_BR_BITS'(alu_dispatch_if[0].data.op_type) == `INST_BR_EBREAK
                      || `INST_BR_BITS'(alu_dispatch_if[0].data.op_type) == `INST_BR_ECALL);

endmodule
