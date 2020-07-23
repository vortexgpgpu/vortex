`include "VX_define.vh"

module VX_issue_mux (
    // inputs
    VX_decode_if    decode_if,
    VX_gpr_data_if  gpr_data_if,

    // outputs
    VX_alu_req_if   alu_req_if,
    VX_lsu_req_if   lsu_req_if,
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,
    VX_fpu_req_if   fpu_req_if,
    VX_gpu_req_if   gpu_req_if    
);

    wire[`NUM_THREADS-1:0] is_alu = {`NUM_THREADS{decode_if.ex_type == `EX_ALU}};
    wire[`NUM_THREADS-1:0] is_lsu = {`NUM_THREADS{decode_if.ex_type == `EX_LSU}};
    wire[`NUM_THREADS-1:0] is_csr = {`NUM_THREADS{decode_if.ex_type == `EX_CSR}};
    wire[`NUM_THREADS-1:0] is_mul = {`NUM_THREADS{decode_if.ex_type == `EX_MUL}};
    wire[`NUM_THREADS-1:0] is_fpu = {`NUM_THREADS{decode_if.ex_type == `EX_FPU}};
    wire[`NUM_THREADS-1:0] is_gpu = {`NUM_THREADS{decode_if.ex_type == `EX_GPU}};

    // ALU unit
    assign alu_req_if.valid       = decode_if.valid & is_alu;
    assign alu_req_if.warp_num    = decode_if.warp_num;
    assign alu_req_if.curr_PC     = decode_if.curr_PC;
    assign alu_req_if.alu_op      = `ALU_OP(decode_if.instr_op);
    assign alu_req_if.rd          = decode_if.rd;
    assign alu_req_if.wb          = decode_if.wb;
    assign alu_req_if.rs1_data    = gpr_data_if.rs1_data;
    assign alu_req_if.rs2_data    = gpr_data_if.rs2_data;    
    assign alu_req_if.offset      = decode_if.imm;
    assign alu_req_if.next_PC     = decode_if.next_PC;

    // LSU unit
    assign lsu_req_if.valid       = decode_if.valid & is_lsu;
    assign lsu_req_if.warp_num    = decode_if.warp_num;
    assign lsu_req_if.curr_PC     = decode_if.curr_PC;
    assign lsu_req_if.base_addr   = gpr_data_if.rs1_data;
    assign lsu_req_if.store_data  = gpr_data_if.rs2_data;
    assign lsu_req_if.offset      = decode_if.imm;
    assign lsu_req_if.rw          = `LSU_RW(decode_if.instr_op);
    assign lsu_req_if.byteen      = `LSU_BE(decode_if.instr_op);
    assign lsu_req_if.rd          = decode_if.rd;
    assign lsu_req_if.wb          = decode_if.wb;    

    // CSR unit
    assign csr_req_if.valid       = decode_if.valid & is_csr;
    assign csr_req_if.warp_num    = decode_if.warp_num;
    assign csr_req_if.curr_PC     = decode_if.curr_PC;
    assign csr_req_if.csr_op      = `CSR_OP(decode_if.instr_op);
    assign csr_req_if.csr_addr    = decode_if.imm[`CSR_ADDR_SIZE-1:0];
    assign csr_req_if.csr_mask    = decode_if.rs2_is_imm ? 32'(decode_if.rs1) : gpr_data_if.rs1_data[0];
    assign csr_req_if.rd          = decode_if.rd;
    assign csr_req_if.wb          = decode_if.wb;
    assign csr_req_if.is_io       = 1'b0;

    // MUL unit
    assign mul_req_if.valid       = decode_if.valid & is_mul;
    assign mul_req_if.warp_num    = decode_if.warp_num;
    assign mul_req_if.curr_PC     = decode_if.curr_PC;
    assign mul_req_if.mul_op      = `MUL_OP(decode_if.instr_op);
    assign mul_req_if.rs1_data    = gpr_data_if.rs1_data;
    assign mul_req_if.rs2_data    = gpr_data_if.rs2_data;    
    assign mul_req_if.rd          = decode_if.rd;
    assign mul_req_if.wb          = decode_if.wb;

    // FPU unit
    assign fpu_req_if.valid       = decode_if.valid & is_fpu;
    assign fpu_req_if.warp_num    = decode_if.warp_num;
    assign fpu_req_if.curr_PC     = decode_if.curr_PC;
    assign fpu_req_if.fpu_op      = `FPU_OP(decode_if.instr_op);
    assign fpu_req_if.rs1_data    = gpr_data_if.rs1_data;
    assign fpu_req_if.rs2_data    = gpr_data_if.rs2_data;    
    assign fpu_req_if.rs3_data    = gpr_data_if.rs3_data;    
    assign fpu_req_if.frm         = decode_if.frm;
    assign fpu_req_if.rd          = decode_if.rd;
    assign fpu_req_if.wb          = decode_if.wb;

    // GPU unit
    assign gpu_req_if.valid       = decode_if.valid & is_gpu;
    assign gpu_req_if.warp_num    = decode_if.warp_num;
    assign gpu_req_if.curr_PC     = decode_if.curr_PC;
    assign gpu_req_if.gpu_op      = `GPU_OP(decode_if.instr_op);
    assign gpu_req_if.rs1_data    = gpr_data_if.rs1_data;
    assign gpu_req_if.rs2_data    = gpr_data_if.rs2_data[0];
    assign gpu_req_if.next_PC     = decode_if.next_PC;
    
endmodule