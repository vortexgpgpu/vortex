`include "VX_define.vh"

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_execute

    input wire clk, 
    input wire reset,    

    // Dcache interface
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // commit status
    VX_cmt_to_csr_if    cmt_to_csr_if,

`ifdef PERF_ENABLE
    VX_perf_memsys_if   perf_memsys_if,
    VX_perf_pipeline_if perf_pipeline_if,
 `endif
    
    // inputs    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,  
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if,
    
    // outputs
    VX_branch_ctl_if    branch_ctl_if,    
    VX_warp_ctl_if      warp_ctl_if,
    VX_commit_if        alu_commit_if,
    VX_commit_if        ld_commit_if,
    VX_commit_if        st_commit_if,
    VX_commit_if        csr_commit_if,
    VX_commit_if        fpu_commit_if,
    VX_commit_if        gpu_commit_if,
    
    input wire          busy
);
    VX_fpu_to_csr_if     fpu_to_csr_if(); 
    wire[`NUM_WARPS-1:0] csr_pending;
    wire[`NUM_WARPS-1:0] fpu_pending;
    
    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .alu_req_if     (alu_req_if),
        .branch_ctl_if  (branch_ctl_if),
        .alu_commit_if  (alu_commit_if)
    );

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_BIND_VX_execute_lsu_unit
        .clk            (clk),
        .reset          (reset),
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
        .lsu_req_if     (lsu_req_if),
        .ld_commit_if   (ld_commit_if),
        .st_commit_if   (st_commit_if)
    );

    VX_csr_unit #(
        .CORE_ID(CORE_ID)
    ) csr_unit (
        .clk            (clk),
        .reset          (reset),   
    `ifdef PERF_ENABLE
        .perf_memsys_if  (perf_memsys_if),
        .perf_pipeline_if (perf_pipeline_if),
    `endif    
        .cmt_to_csr_if  (cmt_to_csr_if),    
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if),
        .fpu_pending    (fpu_pending),
        .pending        (csr_pending),
        .busy           (busy)
    );

`ifdef EXT_F_ENABLE
    VX_fpu_unit #(
        .CORE_ID(CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),        
        .fpu_req_if     (fpu_req_if), 
        .fpu_to_csr_if  (fpu_to_csr_if), 
        .fpu_commit_if  (fpu_commit_if),
        .csr_pending    (csr_pending),
        .pending        (fpu_pending) 
    );
`else
    `UNUSED_VAR (csr_pending)
    `UNUSED_VAR (fpu_to_csr_if.read_frm)
    assign fpu_req_if.ready     = 0;
    assign fpu_commit_if.valid  = 0;
    assign fpu_commit_if.wid    = 0;
    assign fpu_commit_if.PC     = 0;
    assign fpu_commit_if.tmask  = 0;
    assign fpu_commit_if.wb     = 0;
    assign fpu_commit_if.rd     = 0;
    assign fpu_commit_if.data   = 0;  
    assign fpu_to_csr_if.write_enable = 0;  
    assign fpu_to_csr_if.write_wid = 0;
    assign fpu_to_csr_if.write_fflags = 0;
    assign fpu_to_csr_if.read_wid = 0;
    assign fpu_pending = 0;
`endif

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        `SCOPE_BIND_VX_execute_gpu_unit
        .clk            (clk),
        .reset          (reset),    
        .gpu_req_if     (gpu_req_if),
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    // special workaround to get RISC-V tests Pass/Fail status
    wire ebreak /* verilator public */;
    assign ebreak = alu_req_if.valid && alu_req_if.ready
                 && `ALU_IS_BR(alu_req_if.op_mod)
                 && (`BR_OP(alu_req_if.op_type) == `BR_EBREAK 
                  || `BR_OP(alu_req_if.op_type) == `BR_ECALL);

endmodule
