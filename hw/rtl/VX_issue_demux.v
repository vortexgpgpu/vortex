`include "VX_define.vh"

module VX_issue_demux (
    // inputs
    VX_issue_if     issue_if,
    
    // outputs
    VX_alu_req_if   alu_req_if,
    VX_bru_req_if   bru_req_if,
    VX_lsu_req_if   lsu_req_if,
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,
    VX_fpu_req_if   fpu_req_if,
    VX_gpu_req_if   gpu_req_if    
);
    // ALU unit
    assign alu_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_ALU);
    assign alu_req_if.issue_tag   = issue_if.issue_tag;
    assign alu_req_if.wid         = issue_if.wid;
    assign alu_req_if.thread_mask = issue_if.thread_mask;
    assign alu_req_if.curr_PC     = issue_if.curr_PC;
    assign alu_req_if.op          = `ALU_OP(issue_if.ex_op);
    assign alu_req_if.rs1_is_PC   = issue_if.rs1_is_PC;
    assign alu_req_if.rs2_is_imm  = issue_if.rs2_is_imm;
    assign alu_req_if.imm         = issue_if.imm;
    assign alu_req_if.rs1_data    = issue_if.rs1_data;
    assign alu_req_if.rs2_data    = issue_if.rs2_data;

    // BRU unit
    assign bru_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_BRU);
    assign bru_req_if.issue_tag   = issue_if.issue_tag;
    assign bru_req_if.wid         = issue_if.wid;
    assign bru_req_if.thread_mask = issue_if.thread_mask;
    assign bru_req_if.curr_PC     = issue_if.curr_PC;
    assign bru_req_if.op          = `BRU_OP(issue_if.ex_op);
    assign bru_req_if.rs1_is_PC   = issue_if.rs1_is_PC;
    assign bru_req_if.rs1_data    = issue_if.rs1_data[issue_if.tid];  
    assign bru_req_if.rs2_data    = issue_if.rs2_data[issue_if.tid];  
    assign bru_req_if.offset      = issue_if.imm;
    
    // LSU unit
    assign lsu_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_LSU);
    assign lsu_req_if.issue_tag   = issue_if.issue_tag;
    assign lsu_req_if.wid         = issue_if.wid;
    assign lsu_req_if.thread_mask = issue_if.thread_mask;
    assign lsu_req_if.curr_PC     = issue_if.curr_PC;
    assign lsu_req_if.rw          = `LSU_RW(issue_if.ex_op);
    assign lsu_req_if.byteen      = `LSU_BE(issue_if.ex_op);
    assign lsu_req_if.base_addr   = issue_if.rs1_data;
    assign lsu_req_if.store_data  = issue_if.rs2_data;
    assign lsu_req_if.offset      = issue_if.imm;
    assign lsu_req_if.rd          = issue_if.rd;
    assign lsu_req_if.wb          = issue_if.wb;

    // CSR unit
    assign csr_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_CSR);
    assign csr_req_if.issue_tag   = issue_if.issue_tag;
    assign csr_req_if.wid         = issue_if.wid;
    assign csr_req_if.thread_mask = issue_if.thread_mask;
    assign csr_req_if.curr_PC     = issue_if.curr_PC;
    assign csr_req_if.op          = `CSR_OP(issue_if.ex_op);
    assign csr_req_if.csr_addr    = issue_if.imm[`CSR_ADDR_BITS-1:0];
    assign csr_req_if.csr_mask    = issue_if.rs2_is_imm ? 32'(issue_if.rs1) : issue_if.rs1_data[0];
    assign csr_req_if.is_io       = 1'b0;

    // MUL unit
`ifdef EXT_M_ENABLE    
    assign mul_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_MUL);
    assign mul_req_if.issue_tag   = issue_if.issue_tag;
    assign mul_req_if.wid         = issue_if.wid;
    assign mul_req_if.thread_mask = issue_if.thread_mask;
    assign mul_req_if.curr_PC     = issue_if.curr_PC;
    assign mul_req_if.op          = `MUL_OP(issue_if.ex_op);
    assign mul_req_if.rs1_data    = issue_if.rs1_data;
    assign mul_req_if.rs2_data    = issue_if.rs2_data;   
`endif

    // FPU unit
`ifdef EXT_F_ENABLE    
    assign fpu_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_FPU);
    assign fpu_req_if.issue_tag   = issue_if.issue_tag;
    assign fpu_req_if.wid         = issue_if.wid;
    assign fpu_req_if.thread_mask = issue_if.thread_mask;
    assign fpu_req_if.curr_PC     = issue_if.curr_PC;
    assign fpu_req_if.op          = `FPU_OP(issue_if.ex_op);
    assign fpu_req_if.frm         = issue_if.frm;
    assign fpu_req_if.rs1_data    = issue_if.rs1_data;
    assign fpu_req_if.rs2_data    = issue_if.rs2_data;    
    assign fpu_req_if.rs3_data    = issue_if.rs3_data;        
`endif

    // GPU unit
    assign gpu_req_if.valid       = issue_if.valid && (issue_if.ex_type == `EX_GPU);
    assign gpu_req_if.issue_tag   = issue_if.issue_tag;
    assign gpu_req_if.wid         = issue_if.wid;
    assign gpu_req_if.thread_mask = issue_if.thread_mask;
    assign gpu_req_if.curr_PC     = issue_if.curr_PC;
    assign gpu_req_if.op          = `GPU_OP(issue_if.ex_op);
    assign gpu_req_if.rs1_data    = issue_if.rs1_data;
    assign gpu_req_if.rs2_data    = issue_if.rs2_data[0];
    
endmodule