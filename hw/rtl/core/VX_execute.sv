`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk, 
    input wire              reset,    

    input base_dcrs_t       base_dcrs,

    // Dcache interface
    VX_cache_req_if.master  dcache_req_if,
    VX_cache_rsp_if.slave   dcache_rsp_if,

    // commit interface
    VX_cmt_to_csr_if.slave  cmt_to_csr_if,

    // fetch interface
    VX_fetch_to_csr_if.slave fetch_to_csr_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave perf_memsys_if,
    VX_perf_pipeline_if.slave perf_pipeline_if,
`endif

`ifdef EXT_F_ENABLE
    VX_fpu_agent_if.slave   fpu_agent_if,
    VX_fpu_req_if.master    fpu_req_if,
    VX_fpu_rsp_if.slave     fpu_rsp_if,
    VX_commit_if.master     fpu_commit_if,
`endif

`ifdef EXT_TEX_ENABLE
    VX_tex_req_if.master    tex_req_if,
    VX_tex_rsp_if.slave     tex_rsp_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave    perf_tex_if,
    VX_perf_cache_if.slave  perf_tcache_if,
`endif
`endif

`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if.slave  raster_req_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave perf_raster_if,
    VX_perf_cache_if.slave  perf_rcache_if,
`endif
`endif

`ifdef EXT_ROP_ENABLE        
    VX_rop_req_if.master    rop_req_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave    perf_rop_if,
    VX_perf_cache_if.slave  perf_ocache_if,
`endif
`endif    
  
    VX_alu_req_if.slave     alu_req_if,
    VX_branch_ctl_if.master branch_ctl_if,    
    VX_commit_if.master     alu_commit_if,
    
    VX_lsu_req_if.slave     lsu_req_if,    
    VX_commit_if.master     ld_commit_if,
    VX_commit_if.master     st_commit_if,
    
    VX_csr_req_if.slave     csr_req_if,
    VX_commit_if.master     csr_commit_if,
    
    VX_gpu_req_if.slave     gpu_req_if,
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_commit_if.master     gpu_commit_if,    

    // simulation helper signals
    output wire             sim_ebreak
);

    wire gpu_pending;
    wire csr_pending;

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if tex_csr_if();
`endif

`ifdef EXT_RASTER_ENABLE
    VX_gpu_csr_if raster_csr_if();
`endif

`ifdef EXT_ROP_ENABLE
    VX_gpu_csr_if rop_csr_if();
`endif

`ifdef EXT_F_ENABLE
    wire fpu_pending;
    VX_fpu_to_csr_if fpu_to_csr_if();
`endif

`ifdef PERF_ENABLE
    VX_perf_gpu_if perf_gpu_if();
`endif

    `RESET_RELAY (alu_reset, reset);
    `RESET_RELAY (lsu_reset, reset);
    `RESET_RELAY (csr_reset, reset);
    `RESET_RELAY (gpu_reset, reset);
    
    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (alu_reset),
        .alu_req_if     (alu_req_if),
        .branch_ctl_if  (branch_ctl_if),
        .alu_commit_if  (alu_commit_if)
    );

    `SCOPE_IO_SWITCH (1)

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_IO_BIND  (0)
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

        .base_dcrs      (base_dcrs),
    
    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if(perf_pipeline_if),
        .perf_gpu_if    (perf_gpu_if),
    `endif

        .gpu_pending    (gpu_pending),

        .req_pending    (csr_pending),
    
    `ifdef EXT_F_ENABLE  
        .fpu_to_csr_if  (fpu_to_csr_if),
        .fpu_pending    (fpu_pending),
    `endif        
    
    `ifdef EXT_TEX_ENABLE        
        .tex_csr_if     (tex_csr_if),
    `ifdef PERF_ENABLE
        .perf_tex_if    (perf_tex_if),
        .perf_tcache_if (perf_tcache_if),
    `endif
    `endif
    
    `ifdef EXT_RASTER_ENABLE        
        .raster_csr_if  (raster_csr_if),
    `ifdef PERF_ENABLE
        .perf_raster_if (perf_raster_if),
        .perf_rcache_if (perf_rcache_if),
    `endif
    `endif

    `ifdef EXT_ROP_ENABLE        
        .rop_csr_if     (rop_csr_if),
    `ifdef PERF_ENABLE
        .perf_rop_if    (perf_rop_if),
        .perf_ocache_if (perf_ocache_if),
    `endif
    `endif

        .cmt_to_csr_if  (cmt_to_csr_if),
        .fetch_to_csr_if(fetch_to_csr_if),
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if)
    );

`ifdef EXT_F_ENABLE
    `RESET_RELAY (fpu_reset, reset);

    VX_fpu_agent #(
        .CORE_ID(CORE_ID)
    ) fpu_agent (
        .clk            (clk),
        .reset          (fpu_reset),    
        .fpu_agent_if   (fpu_agent_if), 
        .fpu_req_if     (fpu_req_if),
        .fpu_rsp_if     (fpu_rsp_if),
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .fpu_commit_if  (fpu_commit_if),
        .csr_pending    (csr_pending),
        .req_pending    (fpu_pending) 
    );
`endif

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        .clk            (clk),
        .reset          (gpu_reset),    
        .gpu_req_if     (gpu_req_if),

    `ifdef PERF_ENABLE
        .perf_gpu_if    (perf_gpu_if),
    `endif
    
    `ifdef EXT_TEX_ENABLE
        .tex_csr_if     (tex_csr_if),
        .tex_req_if     (tex_req_if),
        .tex_rsp_if     (tex_rsp_if),
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
        .gpu_commit_if  (gpu_commit_if),

        .csr_pending    (csr_pending),
        .req_pending    (gpu_pending) 
    );

    // simulation helper signal to get RISC-V tests Pass/Fail status
    assign sim_ebreak = alu_req_if.valid && alu_req_if.ready
                     && `INST_ALU_IS_BR(alu_req_if.op_mod)
                     && (`INST_BR_BITS'(alu_req_if.op_type) == `INST_BR_EBREAK
                      || `INST_BR_BITS'(alu_req_if.op_type) == `INST_BR_ECALL);

endmodule
