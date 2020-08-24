`include "VX_define.vh"

module VX_execute #(
    parameter CORE_ID = 0
) (
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_BE_IO

    input wire clk, 
    input wire reset, 

    // CSR io interface
    VX_csr_io_req_if    csr_io_req_if,
    VX_csr_io_rsp_if    csr_io_rsp_if,    

    // Dcache interface
    VX_cache_core_req_if dcache_req_if,
    VX_cache_core_rsp_if dcache_rsp_if,

    // perf
    VX_cmt_to_csr_if    cmt_to_csr_if,
    
    // inputs    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if,
    
    // outputs
    VX_csr_to_issue_if  csr_to_issue_if,
    VX_branch_ctl_if    branch_ctl_if,    
    VX_warp_ctl_if      warp_ctl_if,
    VX_exu_to_cmt_if    alu_commit_if,
    VX_exu_to_cmt_if    lsu_commit_if,    
    VX_exu_to_cmt_if    csr_commit_if,
    VX_exu_to_cmt_if    mul_commit_if,
    VX_fpu_to_cmt_if    fpu_commit_if,
    VX_exu_to_cmt_if    gpu_commit_if,
    
    output wire         ebreak
);
    
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
        `SCOPE_SIGNALS_LSU_BIND
        .clk            (clk),
        .reset          (reset),
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
        .lsu_req_if     (lsu_req_if),
        .lsu_commit_if  (lsu_commit_if)
    );

    VX_csr_unit #(
        .CORE_ID(CORE_ID)
    ) csr_unit (
        .clk            (clk),
        .reset          (reset),    
        .cmt_to_csr_if  (cmt_to_csr_if),    
        .csr_to_issue_if  (csr_to_issue_if), 
        .csr_io_req_if  (csr_io_req_if),           
        .csr_io_rsp_if  (csr_io_rsp_if),
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if)
    );

`ifdef EXT_M_ENABLE
    VX_mul_unit #(
        .CORE_ID(CORE_ID)
    ) mul_unit (
        .clk            (clk),
        .reset          (reset),
        .mul_req_if     (mul_req_if),
        .mul_commit_if  (mul_commit_if)    
    );
`else
    assign mul_req_if.ready        = 0;
    assign mul_commit_if.valid     = 0;
    assign mul_commit_if.issue_tag = 0;
    assign mul_commit_if.data      = 0;
`endif

`ifdef EXT_F_ENABLE
    VX_fpu_unit #(
        .CORE_ID(CORE_ID)
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),        
        .fpu_req_if     (fpu_req_if),
        .fpu_commit_if  (fpu_commit_if)    
    );
`else
    assign fpu_req_if.ready         = 0;
    assign fpu_commit_if.valid      = 0;
    assign fpu_commit_if.issue_tag  = 0;
    assign fpu_commit_if.data       = 0;
    assign fpu_commit_if.has_fflags = 0;
    assign fpu_commit_if.fflags     = 0;
`endif

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        .clk            (clk),
        .reset          (reset),    
        .gpu_req_if     (gpu_req_if),
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    assign ebreak = alu_req_if.valid 
                 && alu_req_if.is_br_op
                 && (`BR_OP(alu_req_if.op_type) == `BR_EBREAK 
                  || `BR_OP(alu_req_if.op_type) == `BR_ECALL);

    `SCOPE_ASSIGN (scope_decode_valid,       decode_if.valid);
    `SCOPE_ASSIGN (scope_decode_wid,         decode_if.wid);
    `SCOPE_ASSIGN (scope_decode_curr_PC,     decode_if.curr_PC);    
    `SCOPE_ASSIGN (scope_decode_is_jal,      decode_if.is_jal);
    `SCOPE_ASSIGN (scope_decode_rs1,         decode_if.rs1);
    `SCOPE_ASSIGN (scope_decode_rs2,         decode_if.rs2);

    `SCOPE_ASSIGN (scope_execute_valid,      alu_req_if.valid);    
    `SCOPE_ASSIGN (scope_execute_wid,        alu_req_if.wid);
    `SCOPE_ASSIGN (scope_execute_curr_PC,    alu_req_if.curr_PC);    
    `SCOPE_ASSIGN (scope_execute_rd,         alu_req_if.rd);
    `SCOPE_ASSIGN (scope_execute_a,          alu_req_if.rs1_data);
    `SCOPE_ASSIGN (scope_execute_b,          alu_req_if.rs2_data);   
        
    `SCOPE_ASSIGN (scope_writeback_valid,    writeback_if.valid);    
    `SCOPE_ASSIGN (scope_writeback_wid,      writeback_if.wid);
    `SCOPE_ASSIGN (scope_writeback_curr_PC,  writeback_if.curr_PC);  
    `SCOPE_ASSIGN (scope_writeback_wb,       writeback_if.wb);      
    `SCOPE_ASSIGN (scope_writeback_rd,       writeback_if.rd);
    `SCOPE_ASSIGN (scope_writeback_data,     writeback_if.data);

endmodule
