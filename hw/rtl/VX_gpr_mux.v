`include "VX_define.vh"

module VX_gpr_mux (
    // inputs
    VX_execute_if execute_if,
    input wire [`NUM_THREADS-1:0][31:0] rs1_data,
    input wire [`NUM_THREADS-1:0][31:0] rs2_data,

    // outputs
    VX_alu_req_if   alu_req_if,
    VX_branch_req_if branch_req_if,
    VX_lsu_req_if   lsu_req_if,
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,
    VX_gpu_req_if   gpu_req_if    
);

    wire[`NUM_THREADS-1:0] is_alu = {`NUM_THREADS{execute_if.ex_type == `EX_ALU}};
    wire[`NUM_THREADS-1:0] is_br  = {`NUM_THREADS{execute_if.ex_type == `EX_BR}};    
    wire[`NUM_THREADS-1:0] is_lsu = {`NUM_THREADS{execute_if.ex_type == `EX_LSU}};
    wire[`NUM_THREADS-1:0] is_csr = {`NUM_THREADS{execute_if.ex_type == `EX_CSR}};
    wire[`NUM_THREADS-1:0] is_mul = {`NUM_THREADS{execute_if.ex_type == `EX_MUL}};
    wire[`NUM_THREADS-1:0] is_gpu = {`NUM_THREADS{execute_if.ex_type == `EX_GPU}};

    // ALU unit
    assign alu_req_if.valid       = execute_if.valid & is_alu;
    assign alu_req_if.warp_num    = execute_if.warp_num;
    assign alu_req_if.curr_PC     = execute_if.curr_PC;
    assign alu_req_if.alu_op      = `ALU_OP(execute_if.instr_op);
    assign alu_req_if.rd          = execute_if.rd;
    assign alu_req_if.wb          = execute_if.wb;
    assign alu_req_if.rs1_data    = rs1_data;
    assign alu_req_if.rs2_data    = rs2_data;    

    // BR unit
    assign branch_req_if.valid    = execute_if.valid & is_br;
    assign branch_req_if.warp_num = execute_if.warp_num;
    assign branch_req_if.curr_PC  = execute_if.curr_PC;    
    assign branch_req_if.br_op    = `BR_OP(execute_if.instr_op);
    assign branch_req_if.offset   = execute_if.imm;
    assign branch_req_if.next_PC  = execute_if.next_PC;
    assign branch_req_if.rs1_data = rs1_data;
    assign branch_req_if.rs2_data = rs2_data;    
    assign branch_req_if.rd       = execute_if.rd;
    assign branch_req_if.wb       = execute_if.wb;

    // LSU unit
    assign lsu_req_if.valid       = execute_if.valid & is_lsu;
    assign lsu_req_if.warp_num    = execute_if.warp_num;
    assign lsu_req_if.curr_PC     = execute_if.curr_PC;
    assign lsu_req_if.base_addr   = rs1_data;
    assign lsu_req_if.store_data  = rs2_data;
    assign lsu_req_if.offset      = execute_if.imm;
    assign lsu_req_if.rw          = `LSU_RW(execute_if.instr_op);
    assign lsu_req_if.byteen      = `LSU_BE(execute_if.instr_op);
    assign lsu_req_if.rd          = execute_if.rd;
    assign lsu_req_if.wb          = execute_if.wb;    

    // CSR unit
    assign csr_req_if.valid       = execute_if.valid & is_csr;
    assign csr_req_if.warp_num    = execute_if.warp_num;
    assign csr_req_if.curr_PC     = execute_if.curr_PC;
    assign csr_req_if.csr_op      = `CSR_OP(execute_if.instr_op);
    assign csr_req_if.csr_addr    = execute_if.imm[`CSR_ADDR_SIZE-1:0];
    assign csr_req_if.csr_mask    = execute_if.rs2_is_imm ? 32'(execute_if.rs1) : rs1_data[0];
    assign csr_req_if.rd          = execute_if.rd;
    assign csr_req_if.wb          = execute_if.wb;
    assign csr_req_if.is_io       = 1'b0;

    // MUL unit
    assign mul_req_if.valid       = execute_if.valid & is_mul;
    assign mul_req_if.warp_num    = execute_if.warp_num;
    assign mul_req_if.curr_PC     = execute_if.curr_PC;
    assign mul_req_if.mul_op      = `MUL_OP(execute_if.instr_op);
    assign mul_req_if.rs1_data    = rs1_data;
    assign mul_req_if.rs2_data    = rs2_data;    
    assign mul_req_if.rd          = execute_if.rd;
    assign mul_req_if.wb          = execute_if.wb;

    // GPU unit
    assign gpu_req_if.valid       = execute_if.valid & is_gpu;
    assign gpu_req_if.warp_num    = execute_if.warp_num;
    assign gpu_req_if.next_PC     = execute_if.next_PC;
    assign gpu_req_if.gpu_op      = `GPU_OP(execute_if.instr_op);
    assign gpu_req_if.rs1_data    = rs1_data;
    assign gpu_req_if.rs2_data    = rs2_data[0];
    
endmodule