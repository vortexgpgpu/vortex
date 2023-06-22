`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_fetch #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    input base_dcrs_t       base_dcrs,

    // Icache interface
    VX_cache_req_if.master  icache_req_if,
    VX_cache_rsp_if.slave   icache_rsp_if,

    // inputs
    VX_wrelease_if.slave    wrelease_if,
    VX_join_if.slave        join_if,
    VX_branch_ctl_if.slave  branch_ctl_if,
    VX_warp_ctl_if.slave    warp_ctl_if,    
    
    // outputs
    VX_ifetch_rsp_if.master ifetch_rsp_if,
    VX_gbar_bus_if.master   gbar_bus_if,

    // csr interface
    VX_fetch_to_csr_if.master fetch_to_csr_if,

    // commit interface
    VX_cmt_to_fetch_if.slave cmt_to_fetch_if,

    // Status
    output wire             busy
);

    VX_ifetch_req_if  ifetch_req_if();

    VX_warp_sched #(
        .CORE_ID(CORE_ID)
    ) warp_sched (
        .clk            (clk),
        .reset          (reset),   

        .base_dcrs      (base_dcrs),  

        .warp_ctl_if    (warp_ctl_if),
        .wrelease_if    (wrelease_if),
        .join_if        (join_if),
        .branch_ctl_if  (branch_ctl_if),

        .ifetch_req_if  (ifetch_req_if),
        .gbar_bus_if    (gbar_bus_if),

        .fetch_to_csr_if(fetch_to_csr_if),

        .cmt_to_fetch_if(cmt_to_fetch_if),

        .busy           (busy)
    ); 

    VX_icache_stage #(
        .CORE_ID(CORE_ID)
    ) icache_stage (
        .clk            (clk),
        .reset          (reset),
        
        .icache_req_if  (icache_req_if),
        .icache_rsp_if  (icache_rsp_if),        

        .ifetch_req_if  (ifetch_req_if),
        .ifetch_rsp_if  (ifetch_rsp_if)   
    );

`ifdef DBG_SCOPE_FETCH
    if (CORE_ID == 0) begin
    `ifdef SCOPE
        localparam UUID_WIDTH = `UP(`UUID_BITS);
        wire ifetch_req_fire = ifetch_req_if.valid && ifetch_req_if.ready;
        wire icache_req_fire = icache_req_if.valid && icache_req_if.ready;
        wire icache_rsp_fire = icache_rsp_if.valid && icache_rsp_if.ready;
        VX_scope_tap #(
            .SCOPE_ID (1),
            .TRIGGERW (7),
            .PROBEW   (3*UUID_WIDTH + 237)
        ) scope_tap (
            .clk(clk),
            .reset(scope_reset),
            .start(1'b0),
            .stop(1'b0),
            .triggers({
                reset,
                ifetch_req_fire,
                icache_req_fire,
                icache_rsp_fire,
                warp_ctl_if.valid,
                branch_ctl_if.valid,
                join_if.valid
            }),
            .probes({
                ifetch_req_if.uuid, ifetch_req_if.wid, ifetch_req_if.tmask, ifetch_req_if.PC,
                icache_req_if.tag, icache_req_if.byteen, icache_req_if.addr,
                icache_rsp_if.data, icache_rsp_if.tag,
                join_if.wid, warp_ctl_if.barrier, warp_ctl_if.split, warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.wid,
                branch_ctl_if.dest, branch_ctl_if.taken, branch_ctl_if.wid
            }),
            .bus_in(scope_bus_in),
            .bus_out(scope_bus_out)
        );
    `endif
    `ifdef CHIPSCOPE
        ila_fetch ila_fetch_inst (
            .clk    (clk),
            .probe0 ({reset, ifetch_req_if.uuid, ifetch_req_if.wid, ifetch_req_if.tmask, ifetch_req_if.PC, ifetch_req_if.ready, ifetch_req_if.valid}),        
            .probe1 ({icache_req_if.tag, icache_req_if.byteen, icache_req_if.addr, icache_req_if.ready, icache_req_if.valid}),
            .probe2 ({icache_rsp_if.data, icache_rsp_if.tag, icache_rsp_if.ready, icache_rsp_if.valid}),
            .probe3 ({join_if.wid, join_if.valid, warp_ctl_if.barrier, warp_ctl_if.split, warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.wid, warp_ctl_if.valid}),
            .probe4 ({branch_ctl_if.dest, branch_ctl_if.taken, branch_ctl_if.wid, branch_ctl_if.valid})
        );
    `endif
    end
`else
    `SCOPE_IO_UNUSED()
`endif

endmodule
