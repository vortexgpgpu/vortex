`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_fetch #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_fetch

    input wire clk,
    input wire reset,

    input base_dcrs_t      base_dcrs,

    // Icache interface
    VX_cache_req_if.master icache_req_if,
    VX_cache_rsp_if.slave  icache_rsp_if,

    // inputs
    VX_wrelease_if.slave    wrelease_if,
    VX_join_if.slave        join_if,
    VX_branch_ctl_if.slave  branch_ctl_if,
    VX_warp_ctl_if.slave    warp_ctl_if,

    // outputs
    VX_ifetch_rsp_if.master ifetch_rsp_if,

    // csr interface
    VX_fetch_to_csr_if.master fetch_to_csr_if,

    // Status
    output wire             busy
);

    VX_ifetch_req_if  ifetch_req_if();

    VX_warp_sched #(
        .CORE_ID(CORE_ID)
    ) warp_sched (
        `SCOPE_BIND_VX_fetch_warp_sched

        .clk              (clk),
        .reset            (reset),   

        .base_dcrs        (base_dcrs),  

        .warp_ctl_if      (warp_ctl_if),
        .wrelease_if      (wrelease_if),
        .join_if          (join_if),
        .branch_ctl_if    (branch_ctl_if),

        .ifetch_req_if    (ifetch_req_if),

        .fetch_to_csr_if  (fetch_to_csr_if),

        .busy             (busy)
    ); 

    VX_icache_stage #(
        .CORE_ID(CORE_ID)
    ) icache_stage (
        `SCOPE_BIND_VX_fetch_icache_stage

        .clk            (clk),
        .reset          (reset),
        
        .icache_rsp_if  (icache_rsp_if),
        .icache_req_if  (icache_req_if),

        .ifetch_req_if  (ifetch_req_if),
        .ifetch_rsp_if  (ifetch_rsp_if)   
    );

endmodule
