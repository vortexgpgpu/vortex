`include "VX_define.vh"

module VX_issue_demux (
    // inputs
    VX_decode_if    decode_if,
    VX_gpr_read_if  gpr_read_if,
    input wire [`ISTAG_BITS-1:0] issue_tag,

    // outputs
    VX_alu_req_if   alu_req_if,
    VX_lsu_req_if   lsu_req_if,
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,
    VX_fpu_req_if   fpu_req_if,
    VX_gpu_req_if   gpu_req_if    
);
    // ALU unit
    assign alu_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_ALU);
    assign alu_req_if.thread_mask = decode_if.thread_mask;
    assign alu_req_if.issue_tag   = issue_tag;
    assign alu_req_if.warp_num    = decode_if.warp_num;
    assign alu_req_if.curr_PC     = decode_if.curr_PC;
    assign alu_req_if.alu_op      = `ALU_OP(decode_if.ex_op);
    assign alu_req_if.rs1_data    = decode_if.rs1_is_PC  ? {`NUM_THREADS{decode_if.curr_PC}} : gpr_read_if.rs1_data;
    assign alu_req_if.rs2_data    = decode_if.rs2_is_imm ? {`NUM_THREADS{decode_if.imm}}     : gpr_read_if.rs2_data;    
    assign alu_req_if.offset      = decode_if.imm;
    assign alu_req_if.next_PC     = decode_if.next_PC;
    
    // LSU unit
    assign lsu_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_LSU);
    assign lsu_req_if.thread_mask = decode_if.thread_mask;
    assign lsu_req_if.issue_tag   = issue_tag;
    assign lsu_req_if.warp_num    = decode_if.warp_num;
    assign lsu_req_if.curr_PC     = decode_if.curr_PC;
    assign lsu_req_if.base_addr   = gpr_read_if.rs1_data;
    assign lsu_req_if.store_data  = gpr_read_if.rs2_data;
    assign lsu_req_if.offset      = decode_if.imm;
    assign lsu_req_if.rw          = `LSU_RW(decode_if.ex_op);
    assign lsu_req_if.byteen      = `LSU_BE(decode_if.ex_op);
    assign lsu_req_if.rd          = decode_if.rd;
    assign lsu_req_if.wb          = decode_if.wb;

    // CSR unit
    assign csr_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_CSR);
    assign csr_req_if.issue_tag   = issue_tag;
    assign csr_req_if.warp_num    = decode_if.warp_num;
    assign csr_req_if.curr_PC     = decode_if.curr_PC;
    assign csr_req_if.csr_op      = `CSR_OP(decode_if.ex_op);
    assign csr_req_if.csr_addr    = decode_if.imm[`CSR_ADDR_SIZE-1:0];
    assign csr_req_if.csr_mask    = decode_if.rs2_is_imm ? 32'(decode_if.rs1) : gpr_read_if.rs1_data[0];
    assign csr_req_if.is_io       = 1'b0;

    // MUL unit
    assign mul_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_MUL);
    assign mul_req_if.issue_tag   = issue_tag;
    assign mul_req_if.mul_op      = `MUL_OP(decode_if.ex_op);
    assign mul_req_if.rs1_data    = gpr_read_if.rs1_data;
    assign mul_req_if.rs2_data    = gpr_read_if.rs2_data;   

    // FPU unit
    assign fpu_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_FPU);
    assign fpu_req_if.issue_tag   = issue_tag;
    assign fpu_req_if.warp_num    = decode_if.warp_num;
    assign fpu_req_if.fpu_op      = `FPU_OP(decode_if.ex_op);
    assign fpu_req_if.rs1_data    = gpr_read_if.rs1_data;
    assign fpu_req_if.rs2_data    = gpr_read_if.rs2_data;    
    assign fpu_req_if.rs3_data    = gpr_read_if.rs3_data;    
    assign fpu_req_if.frm         = decode_if.frm;

    // GPU unit
    assign gpu_req_if.valid       = decode_if.valid && (decode_if.ex_type == `EX_GPU);
    assign gpu_req_if.thread_mask = decode_if.thread_mask;
    assign gpu_req_if.issue_tag   = issue_tag;
    assign gpu_req_if.warp_num    = decode_if.warp_num;
    assign gpu_req_if.gpu_op      = `GPU_OP(decode_if.ex_op);
    assign gpu_req_if.rs1_data    = gpr_read_if.rs1_data;
    assign gpu_req_if.rs2_data    = gpr_read_if.rs2_data[0];
    assign gpu_req_if.next_PC     = decode_if.next_PC;
    
endmodule