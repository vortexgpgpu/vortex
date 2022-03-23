`include "VX_define.vh"

module VX_pipeline #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_pipeline
    
    // Clock
    input wire                  clk,
    input wire                  reset,

    // Dcache interface
    VX_cache_req_if.master     dcache_req_if,
    VX_cache_rsp_if.slave      dcache_rsp_if,

    // Icache interface
    VX_cache_req_if.master     icache_req_if,
    VX_cache_rsp_if.slave      icache_rsp_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave        tex_dcr_if,
    VX_cache_req_if.master     tcache_req_if,
    VX_cache_rsp_if.slave      tcache_rsp_if,
`endif

`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if            raster_req_if,
`endif
`ifdef EXT_RASTER_ENABLE        
    VX_rop_req_if               rop_req_if,
`endif

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave     perf_memsys_if,
`endif

    // simulation helper signals
    output wire                 sim_ebreak,
    output wire [`NUM_REGS-1:0][31:0] sim_last_wb_value,

    // Status
    output wire                 busy
);
    VX_fetch_to_csr_if  fetch_to_csr_if();
    VX_cmt_to_csr_if    cmt_to_csr_if();
    VX_decode_if        decode_if();
    VX_branch_ctl_if    branch_ctl_if();
    VX_warp_ctl_if      warp_ctl_if();
    VX_ifetch_rsp_if    ifetch_rsp_if();
    VX_alu_req_if       alu_req_if();
    VX_lsu_req_if       lsu_req_if();
    VX_csr_req_if       csr_req_if();
`ifdef EXT_F_ENABLE 
    VX_fpu_req_if       fpu_req_if(); 
`endif
    VX_gpu_req_if       gpu_req_if();
    VX_writeback_if     writeback_if();     
    VX_wstall_if        wstall_if();
    VX_join_if          join_if();
    VX_commit_if        alu_commit_if();
    VX_commit_if        ld_commit_if();
    VX_commit_if        st_commit_if();
    VX_commit_if        csr_commit_if();  
`ifdef EXT_F_ENABLE
    VX_commit_if        fpu_commit_if();     
`endif
    VX_commit_if        gpu_commit_if();     

`ifdef PERF_ENABLE
    VX_perf_pipeline_if perf_pipeline_if();
`endif

    `RESET_RELAY (fetch_reset);
    `RESET_RELAY (decode_reset);
    `RESET_RELAY (issue_reset);
    `RESET_RELAY (execute_reset);
    `RESET_RELAY (commit_reset);

    VX_fetch #(
        .CORE_ID(CORE_ID)
    ) fetch (
        `SCOPE_BIND_VX_pipeline_fetch
        .clk            (clk),
        .reset          (fetch_reset),
        .icache_req_if  (icache_req_if),
        .icache_rsp_if  (icache_rsp_if), 
        .wstall_if      (wstall_if),
        .join_if        (join_if),        
        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),
        .ifetch_rsp_if  (ifetch_rsp_if),
        .fetch_to_csr_if(fetch_to_csr_if),
        .busy           (busy)
    );

    VX_decode #(
        .CORE_ID(CORE_ID)
    ) decode (
        .clk            (clk),
        .reset          (decode_reset),        
    `ifdef PERF_ENABLE
        .perf_decode_if (perf_pipeline_if.decode),
    `endif
        .ifetch_rsp_if  (ifetch_rsp_if),
        .decode_if      (decode_if),
        .wstall_if      (wstall_if),
        .join_if        (join_if)
    );

    VX_issue #(
        .CORE_ID(CORE_ID)
    ) issue (
        `SCOPE_BIND_VX_pipeline_issue

        .clk            (clk),
        .reset          (issue_reset),

    `ifdef PERF_ENABLE
        .perf_issue_if  (perf_pipeline_if.issue),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),

        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
    `ifdef EXT_F_ENABLE
        .fpu_req_if     (fpu_req_if),
    `endif
        .gpu_req_if     (gpu_req_if)
    );

    VX_execute #(
        .CORE_ID(CORE_ID)
    ) execute (
        `SCOPE_BIND_VX_pipeline_execute
        
        .clk            (clk),
        .reset          (execute_reset),

    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if (perf_pipeline_if),
    `endif 

        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),

    `ifdef EXT_TEX_ENABLE
        .tex_dcr_if     (tex_dcr_if),
        .tcache_req_if  (tcache_req_if),
        .tcache_rsp_if  (tcache_rsp_if),
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .raster_req_if  (raster_req_if),
    `endif
    `ifdef EXT_RASTER_ENABLE        
        .rop_req_if     (rop_req_if),
    `endif

        .cmt_to_csr_if  (cmt_to_csr_if),   
        .fetch_to_csr_if(fetch_to_csr_if),              
        
        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
    `ifdef EXT_F_ENABLE
        .fpu_req_if     (fpu_req_if),
    `endif
        .gpu_req_if     (gpu_req_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),        
        .alu_commit_if  (alu_commit_if),
        .ld_commit_if   (ld_commit_if),        
        .st_commit_if   (st_commit_if),       
        .csr_commit_if  (csr_commit_if),
    `ifdef EXT_F_ENABLE
        .fpu_commit_if  (fpu_commit_if),
    `endif
        .gpu_commit_if  (gpu_commit_if),

        .sim_ebreak     (sim_ebreak)
    );    

    VX_commit #(
        .CORE_ID(CORE_ID)
    ) commit (
        .clk            (clk),
        .reset          (commit_reset),

        .alu_commit_if  (alu_commit_if),
        .ld_commit_if   (ld_commit_if),        
        .st_commit_if   (st_commit_if),
        .csr_commit_if  (csr_commit_if),
    `ifdef EXT_F_ENABLE
        .fpu_commit_if  (fpu_commit_if),
    `endif
        .gpu_commit_if  (gpu_commit_if),
        
        .writeback_if   (writeback_if),
        .cmt_to_csr_if  (cmt_to_csr_if),

        .sim_last_wb_value (sim_last_wb_value)
    );
    
endmodule
