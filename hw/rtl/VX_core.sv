`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_VX_pipeline
    
    // Clock
    input wire              clk,
    input wire              reset,

    VX_dcr_base_if.slave    dcr_base_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave perf_memsys_if,
`endif

    VX_cache_req_if.master  dcache_req_if,
    VX_cache_rsp_if.slave   dcache_rsp_if,

    VX_cache_req_if.master  icache_req_if,
    VX_cache_rsp_if.slave   icache_rsp_if,    

`ifdef EXT_F_ENABLE
    VX_fpu_req_if.master    fpu_req_if,
    VX_fpu_rsp_if.slave     fpu_rsp_if,
`endif

`ifdef EXT_TEX_ENABLE
    VX_tex_req_if.master    tex_req_if,
    VX_tex_rsp_if.slave     tex_rsp_if,
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave    tex_perf_if,
    VX_perf_cache_if.slave  perf_tcache_if,
`endif
`endif

`ifdef EXT_RASTER_ENABLE        
    VX_raster_req_if.slave  raster_req_if,
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave raster_perf_if,
    VX_perf_cache_if.slave  perf_rcache_if,
`endif
`endif

`ifdef EXT_ROP_ENABLE
    VX_rop_req_if.master    rop_req_if,
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave    rop_perf_if,
    VX_perf_cache_if.slave  perf_ocache_if,
`endif
`endif

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave perf_memsys_if,
`endif

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][31:0] sim_wb_value,

    // Status
    output wire             busy
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
    VX_fpu_agent_if     fpu_agent_if();
`endif
    VX_gpu_req_if       gpu_req_if();
    VX_writeback_if     writeback_if();     
    VX_wrelease_if      wrelease_if();
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

    base_dcrs_t base_dcrs;
    always @(posedge clk) begin
        base_dcrs <= dcr_base_if.data;
    end

    `RESET_RELAY (fetch_reset, reset);
    `RESET_RELAY (decode_reset, reset);
    `RESET_RELAY (issue_reset, reset);
    `RESET_RELAY (execute_reset, reset);
    `RESET_RELAY (commit_reset, reset);

    VX_fetch #(
        .CORE_ID(CORE_ID)
    ) fetch (
        `SCOPE_BIND_VX_pipeline_fetch
        .clk            (clk),
        .base_dcrs      (base_dcrs),
        .reset          (fetch_reset),
        .icache_req_if  (icache_req_if),
        .icache_rsp_if  (icache_rsp_if), 
        .wrelease_if    (wrelease_if),
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
        .wrelease_if    (wrelease_if),
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
        .fpu_agent_if   (fpu_agent_if),
    `endif
        .gpu_req_if     (gpu_req_if)
    );

    VX_execute #(
        .CORE_ID(CORE_ID)
    ) execute (
        `SCOPE_BIND_VX_pipeline_execute
        
        .clk            (clk),
        .reset          (execute_reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .perf_memsys_if (perf_memsys_if),
        .perf_pipeline_if(perf_pipeline_if),
    `endif 

        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_agent_if   (fpu_agent_if),
        .fpu_req_if     (fpu_req_if),
        .fpu_rsp_if     (fpu_rsp_if),
        .fpu_commit_if  (fpu_commit_if),
    `endif   

    `ifdef EXT_TEX_ENABLE
        .tex_req_if     (tex_req_if),
        .tex_rsp_if     (tex_rsp_if),
    `ifdef PERF_ENABLE
        .tex_perf_if    (tex_perf_if),
        .perf_tcache_if (perf_tcache_if),
    `endif
    `endif
    
    `ifdef EXT_RASTER_ENABLE        
        .raster_req_if  (raster_req_if),
    `ifdef PERF_ENABLE
        .raster_perf_if (raster_perf_if),
        .perf_rcache_if (perf_rcache_if),
    `endif
    `endif

    `ifdef EXT_ROP_ENABLE        
        .rop_req_if     (rop_req_if),
    `ifdef PERF_ENABLE
        .rop_perf_if    (rop_perf_if),
        .perf_ocache_if (perf_ocache_if),
    `endif
    `endif

        .cmt_to_csr_if  (cmt_to_csr_if),   
        .fetch_to_csr_if(fetch_to_csr_if),              
        
        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
        .gpu_req_if     (gpu_req_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),        
        .alu_commit_if  (alu_commit_if),
        .ld_commit_if   (ld_commit_if),        
        .st_commit_if   (st_commit_if),       
        .csr_commit_if  (csr_commit_if),
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

        .sim_wb_value   (sim_wb_value)
    );
    
endmodule
