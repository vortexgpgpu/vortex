`include "VX_define.vh"

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_execute

    input wire clk, 
    input wire reset,    

    // Dcache interface
    VX_cache_req_if.master dcache_req_if,
    VX_cache_rsp_if.slave dcache_rsp_if,

    VX_dcr_base_if        dcr_base_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave    tex_dcr_if,
    VX_cache_req_if.master tcache_req_if,
    VX_cache_rsp_if.slave  tcache_rsp_if,
`ifdef PERF_ENABLE
    VX_perf_cache_if.slave perf_tcache_if,
`endif
`endif

    // commit interface
    VX_cmt_to_csr_if.slave  cmt_to_csr_if,

    // fetch interface
    VX_fetch_to_csr_if.slave fetch_to_csr_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave perf_memsys_if,
    VX_perf_pipeline_if.slave perf_pipeline_if,
 `endif

`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if        raster_req_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave raster_perf_if,
`endif
`endif
`ifdef EXT_ROP_ENABLE        
    VX_rop_req_if           rop_req_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave    rop_perf_if,
    VX_perf_cache_if.slave  ocache_perf_if,
`endif
`endif
    
    // inputs    
    VX_alu_req_if.slave     alu_req_if,
    VX_lsu_req_if.slave     lsu_req_if,    
    VX_csr_req_if.slave     csr_req_if,  
`ifdef EXT_F_ENABLE
    VX_fpu_req_if.slave     fpu_req_if,    
`endif
    VX_gpu_req_if.slave     gpu_req_if,
    
    // outputs
    VX_branch_ctl_if.master branch_ctl_if,    
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_commit_if.master     alu_commit_if,
    VX_commit_if.master     ld_commit_if,
    VX_commit_if.master     st_commit_if,
    VX_commit_if.master     csr_commit_if,
`ifdef EXT_F_ENABLE
    VX_commit_if.master     fpu_commit_if,
`endif
    VX_commit_if.master     gpu_commit_if,

    // simulation helper signals
    output wire             sim_ebreak
);

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if tex_csr_if();
`ifdef PERF_ENABLE
    VX_tex_perf_if tex_perf_if();
`endif
`endif
`ifdef EXT_RASTER_ENABLE
    VX_gpu_csr_if raster_csr_if();
`endif
`ifdef EXT_ROP_ENABLE
    VX_gpu_csr_if rop_csr_if();
`endif

`ifdef EXT_F_ENABLE
    wire [`NUM_WARPS-1:0] csr_pending;
    wire [`NUM_WARPS-1:0] fpu_pending;
    VX_fpu_to_csr_if fpu_to_csr_if();
`endif

    `RESET_RELAY (alu_reset);
    `RESET_RELAY (lsu_reset);
    `RESET_RELAY (csr_reset);
    `RESET_RELAY (gpu_reset);
    
    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (alu_reset),
        .alu_req_if     (alu_req_if),
        .branch_ctl_if  (branch_ctl_if),
        .alu_commit_if  (alu_commit_if)
    );

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_BIND_VX_execute_lsu_unit
        .clk            (clk),
        .reset          (lsu_reset),
        .cache_req_if   (dcache_req_if),
        .cache_rsp_if   (dcache_rsp_if),
        .lsu_req_if     (lsu_req_if),
        .ld_commit_if   (ld_commit_if),
        .st_commit_if   (st_commit_if)
    );

    VX_csr_unit #(
        .CORE_ID(CORE_ID)
    ) csr_unit (
        .clk            (clk),
        .reset          (csr_reset),   
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if(perf_pipeline_if),
    `endif
    `ifdef EXT_F_ENABLE  
        .fpu_to_csr_if  (fpu_to_csr_if),
        .fpu_pending    (fpu_pending),        
        .req_pending    (csr_pending),
    `else
        `UNUSED_PIN (req_pending),
    `endif
        .dcr_base_if    (dcr_base_if),
    `ifdef EXT_TEX_ENABLE        
        .tex_csr_if     (tex_csr_if),
    `ifdef PERF_ENABLE
        .tex_perf_if    (tex_perf_if),
        .perf_tcache_if (perf_tcache_if),
    `endif
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .raster_csr_if  (raster_csr_if),
    `ifdef PERF_ENABLE
        .raster_perf_if (raster_perf_if),
    `endif
    `endif
    `ifdef EXT_ROP_ENABLE        
        .rop_csr_if     (rop_csr_if),
    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
        .ocache_perf_if (ocache_perf_if),
    `endif
    `endif
        .cmt_to_csr_if  (cmt_to_csr_if),
        .fetch_to_csr_if(fetch_to_csr_if),
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if)
    );

`ifdef EXT_F_ENABLE
    `RESET_RELAY (fpu_reset);

    VX_fpu_unit #(
        .CORE_ID(CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (fpu_reset),        
        .fpu_req_if     (fpu_req_if), 
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .fpu_commit_if  (fpu_commit_if),
        .csr_pending    (csr_pending),
        .req_pending    (fpu_pending) 
    );
`endif

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        `SCOPE_BIND_VX_execute_gpu_unit
        .clk            (clk),
        .reset          (gpu_reset),    
        .gpu_req_if     (gpu_req_if),
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
        .tex_dcr_if     (tex_dcr_if),
        .tcache_req_if  (tcache_req_if),
        .tcache_rsp_if  (tcache_rsp_if),
    `ifdef PERF_ENABLE
        .tex_perf_if    (tex_perf_if),
    `endif
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .raster_csr_if  (raster_csr_if),
        .raster_req_if  (raster_req_if),
    `endif
    `ifdef EXT_ROP_ENABLE        
        .rop_csr_if     (rop_csr_if),
        .rop_req_if     (rop_req_if),
    `endif
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    // simulation helper signal to get RISC-V tests Pass/Fail status
    assign sim_ebreak = alu_req_if.valid && alu_req_if.ready
                 && `INST_ALU_IS_BR(alu_req_if.op_mod)
                 && (`INST_BR_BITS'(alu_req_if.op_type) == `INST_BR_EBREAK 
                  || `INST_BR_BITS'(alu_req_if.op_type) == `INST_BR_ECALL);

endmodule
