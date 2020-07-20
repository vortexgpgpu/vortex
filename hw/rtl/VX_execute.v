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
    VX_perf_cntrs_if    perf_cntrs_if,

    // inputs    
    VX_alu_req_if       alu_req_if,
    VX_branch_req_if    branch_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_gpu_req_if       gpu_req_if,
    
    // outputs
    VX_branch_ctl_if    branch_ctl_if,    
    VX_warp_ctl_if      warp_ctl_if,
    VX_commit_if        alu_commit_if,
    VX_commit_if        branch_commit_if,
    VX_commit_if        lsu_commit_if,    
    VX_commit_if        csr_commit_if,
    VX_commit_if        mul_commit_if,
    VX_commit_if        gpu_commit_if,
    
    output wire         ebreak
);

    VX_alu_unit #(
        .CORE_ID(CORE_ID)
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .alu_req_if     (alu_req_if),
        .alu_commit_if  (alu_commit_if)
    );

    VX_branch_unit #(
        .CORE_ID(CORE_ID)
    ) branch_unit (
        .clk            (clk),
        .reset          (reset),
        .branch_req_if  (branch_req_if),        
        .branch_ctl_if  (branch_ctl_if),
        .branch_commit_if(branch_commit_if)
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

    VX_csr_pipe #(
        .CORE_ID(CORE_ID)
    ) csr_pipe (
        .clk            (clk),
        .reset          (reset),    
        .perf_cntrs_if  (perf_cntrs_if),    
        .csr_io_req_if  (csr_io_req_if),           
        .csr_io_rsp_if  (csr_io_rsp_if),
        .csr_req_if     (csr_req_if),   
        .csr_commit_if  (csr_commit_if)
    );

    VX_mul_unit #(
        .CORE_ID(CORE_ID)
    ) mul_unit (
        .clk            (clk),
        .reset          (reset),
        .mul_req_if     (mul_req_if),
        .mul_commit_if  (mul_commit_if)    
    );

    VX_gpu_unit #(
        .CORE_ID(CORE_ID)
    ) gpu_unit (
        .gpu_req_if     (gpu_req_if),
        .warp_ctl_if    (warp_ctl_if),
        .gpu_commit_if  (gpu_commit_if)
    );

    assign ebreak = (| branch_req_if.valid) && (branch_req_if.br_op == `BR_EBREAK || branch_req_if.br_op == `BR_ECALL);

    `SCOPE_ASSIGN(scope_decode_valid,       decode_if.valid);
    `SCOPE_ASSIGN(scope_decode_warp_num,    decode_if.warp_num);
    `SCOPE_ASSIGN(scope_decode_curr_PC,     decode_if.curr_PC);    
    `SCOPE_ASSIGN(scope_decode_is_jal,      decode_if.is_jal);
    `SCOPE_ASSIGN(scope_decode_rs1,         decode_if.rs1);
    `SCOPE_ASSIGN(scope_decode_rs2,         decode_if.rs2);

    `SCOPE_ASSIGN(scope_execute_valid,      alu_req_if.valid);    
    `SCOPE_ASSIGN(scope_execute_warp_num,   alu_req_if.warp_num);
    `SCOPE_ASSIGN(scope_execute_curr_PC,    alu_req_if.curr_PC);    
    `SCOPE_ASSIGN(scope_execute_rd,         alu_req_if.rd);
    `SCOPE_ASSIGN(scope_execute_a,          alu_req_if.rs1_data);
    `SCOPE_ASSIGN(scope_execute_b,          alu_req_if.rs2_data);   
        
    `SCOPE_ASSIGN(scope_writeback_valid,    writeback_if.valid);    
    `SCOPE_ASSIGN(scope_writeback_warp_num, writeback_if.warp_num);
    `SCOPE_ASSIGN(scope_writeback_curr_PC,  writeback_if.curr_PC);  
    `SCOPE_ASSIGN(scope_writeback_wb,       writeback_if.wb);      
    `SCOPE_ASSIGN(scope_writeback_rd,       writeback_if.rd);
    `SCOPE_ASSIGN(scope_writeback_data,     writeback_if.data);

endmodule
