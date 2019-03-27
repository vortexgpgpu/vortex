
module Vortex(
	input  wire       clk,
	input  wire       reset,
	input  wire[31:0] fe_instruction,
	input  wire[31:0] in_cache_driver_out_data,
	output wire[31:0] curr_PC,
	output wire[31:0] out_cache_driver_in_address,
	output wire[2:0]  out_cache_driver_in_mem_read,
	output wire[2:0]  out_cache_driver_in_mem_write,
	output wire       out_cache_driver_in_valid,
	output wire[31:0] out_cache_driver_in_data
	);


assign curr_PC = fetch_curr_PC;

// From fetch
wire[31:0] fetch_instruction;
wire       fetch_delay;
wire[31:0] fetch_curr_PC;
wire       fetch_valid;

// From f_d_register
wire[31:0] f_d_instruction;
wire[31:0] f_d_curr_PC;
wire       f_d_valid;

// From decode
wire       decode_branch_stall;
wire[11:0] decode_csr_address;
wire       decode_is_csr;
wire[31:0] decode_csr_mask; 
wire[4:0]  decode_rd;
wire[4:0]  decode_rs1;
wire[4:0]  decode_rs2;
wire[31:0] decode_reg_data[1:0];
wire[1:0]  decode_wb;
wire[4:0]  decode_alu_op;
wire       decode_rs2_src; 
reg[31:0]  decode_itype_immed; 
wire[2:0]  decode_mem_read; 
wire[2:0]  decode_mem_write; 
reg[2:0]   decode_branch_type;
reg        decode_jal;
reg[31:0]  decode_jal_offset;
reg[19:0]  decode_upper_immed;
wire[31:0] decode_PC_next;
wire       decode_valid;

// From d_e_register
wire[11:0] d_e_csr_address;
wire       d_e_is_csr; 
wire[31:0] d_e_csr_mask; 
wire[4:0]  d_e_rd;
wire[4:0]  d_e_rs1;
wire[4:0]  d_e_rs2;
wire[31:0] d_e_reg_data[1:0];
wire[4:0]  d_e_alu_op;
wire[1:0]  d_e_wb;
wire       d_e_rs2_src;
wire[31:0] d_e_itype_immed;
wire[2:0]  d_e_mem_read;
wire[2:0]  d_e_mem_write;
wire[2:0]  d_e_branch_type;
wire[19:0] d_e_upper_immed;
wire[31:0] d_e_curr_PC;
wire       d_e_jal;
wire[31:0] d_e_jal_offset;
wire[31:0] d_e_PC_next;
wire       d_e_valid;


// From execute
wire       execute_branch_stall;
wire[11:0] execute_csr_address;
wire       execute_is_csr;
reg[31:0]  execute_csr_result;
reg[31:0]  execute_alu_result;
wire[4:0]  execute_rd;
wire[1:0]  execute_wb;
wire[4:0]  execute_rs1;
wire[4:0]  execute_rs2;
wire[31:0] execute_reg_data[1:0];
wire[2:0]  execute_mem_read;
wire[2:0]  execute_mem_write;
wire       execute_jal;
wire[31:0] execute_jal_dest;
wire[31:0] execute_branch_offset;
wire[31:0] execute_PC_next;
wire       execute_valid;


// From e_m_register
wire       e_m_jal;
wire[31:0] e_m_jal_dest;
wire[11:0] e_m_csr_address;
wire       e_m_is_csr;
wire[31:0] e_m_csr_result;
wire[31:0] e_m_alu_result;
wire[4:0]  e_m_rd;
wire[1:0]  e_m_wb;
wire[4:0]  e_m_rs1;
/* verilator lint_off UNUSED */
wire[31:0] e_m_reg_data[1:0];
/* verilator lint_on UNUSED */
wire[4:0]  e_m_rs2;
wire[2:0]  e_m_mem_read;
wire[2:0]  e_m_mem_write;
wire[31:0] e_m_curr_PC;
wire[31:0] e_m_branch_offset;
wire[2:0]  e_m_branch_type;
wire[31:0] e_m_PC_next;
wire       e_m_valid;


// From memory
wire       memory_delay;
wire       memory_branch_dir;
wire[31:0] memory_branch_dest;
wire[31:0] memory_alu_result;
wire[31:0] memory_mem_result;
wire[4:0]  memory_rd;
wire[1:0]  memory_wb;
wire[4:0]  memory_rs1;
wire[4:0]  memory_rs2;
wire[31:0] memory_PC_next;
wire       memory_valid;

// From m_w_register
wire[31:0] m_w_alu_result;
wire[31:0] m_w_mem_result;
wire[4:0]  m_w_rd;
wire[1:0]  m_w_wb;
/* verilator lint_off UNUSED */
wire[4:0]  m_w_rs1;
wire[4:0]  m_w_rs2;
/* verilator lint_on UNUSED */
wire[31:0] m_w_PC_next;
wire       m_w_valid;

// From writeback
wire[31:0] writeback_write_data;
wire[4:0]  writeback_rd;
wire[1:0]  writeback_wb;

// From csr handler
wire[31:0]  csr_decode_csr_data;


// From forwarding
wire       forwarding_fwd_stall;
wire       forwarding_src1_fwd;
wire       forwarding_src2_fwd;
/* verilator lint_off UNUSED */
wire       forwarding_csr_fwd;
wire[31:0] forwarding_csr_fwd_data;
/* verilator lint_on UNUSED */
wire[31:0] forwarding_src1_fwd_data;
wire[31:0] forwarding_src2_fwd_data;


// Internal
wire total_freeze;
wire interrupt;
wire debug;

assign debug        = 1'b0;
assign interrupt    = 1'b0;
assign total_freeze = fetch_delay || memory_delay;


VX_fetch vx_fetch(
		.clk(clk),
		.reset(reset),
		.in_branch_dir(memory_branch_dir),
		.in_freeze(total_freeze),
		.in_branch_dest(memory_branch_dest),
		.in_branch_stall(decode_branch_stall),
		.in_fwd_stall(forwarding_fwd_stall),
		.in_branch_stall_exe(execute_branch_stall),
		.in_jal(e_m_jal),
		.in_jal_dest(e_m_jal_dest),
		.in_interrupt(interrupt),
		.in_debug(debug),
		.in_instruction(fe_instruction),

		.out_instruction(fetch_instruction),
		.out_delay(fetch_delay),
		.out_curr_PC(fetch_curr_PC),
		.out_valid(fetch_valid)
	);


VX_f_d_reg vx_f_d_reg(
		.clk(clk),
		.reset(reset),
		.in_instruction(fetch_instruction),
		.in_valid(fetch_valid),
		.in_curr_PC(fetch_curr_PC),
		.in_fwd_stall(forwarding_fwd_stall),
		.in_freeze(total_freeze),
		.out_instruction(f_d_instruction),
		.out_curr_PC(f_d_curr_PC),
		.out_valid(f_d_valid)
	);


VX_decode vx_decode(
		.clk(clk),
		.in_instruction(f_d_instruction),
		.in_curr_PC(f_d_curr_PC),
		.in_valid(f_d_valid),
		.in_write_data(writeback_write_data),
		.in_rd(writeback_rd),
		.in_wb(writeback_wb),
		.in_wb_valid(m_w_valid),
		.in_src1_fwd(forwarding_src1_fwd),
		.in_src1_fwd_data(forwarding_src1_fwd_data),
		.in_src2_fwd(forwarding_src2_fwd),
		.in_src2_fwd_data(forwarding_src2_fwd_data),

		.out_csr_address(decode_csr_address),
		.out_is_csr(decode_is_csr),
		.out_csr_mask(decode_csr_mask),

		.out_rd(decode_rd),
		.out_rs1(decode_rs1),
		.out_rs2(decode_rs2),
		.out_reg_data(decode_reg_data),
		.out_wb(decode_wb),
		.out_alu_op(decode_alu_op),
		.out_rs2_src(decode_rs2_src),
		.out_itype_immed(decode_itype_immed),
		.out_mem_read(decode_mem_read),
		.out_mem_write(decode_mem_write),
		.out_branch_type(decode_branch_type),
		.out_branch_stall(decode_branch_stall),
		.out_jal(decode_jal),
		.out_jal_offset(decode_jal_offset),
		.out_upper_immed(decode_upper_immed),
		.out_PC_next(decode_PC_next),
		.out_valid(decode_valid)
	);


VX_d_e_reg vx_d_e_reg(
		.clk(clk),
		.in_rd(decode_rd),
		.in_rs1(decode_rs1),
		.in_rs2(decode_rs2),
		.in_reg_data(decode_reg_data),
		.in_alu_op(decode_alu_op),
		.in_wb(decode_wb),
		.in_rs2_src(decode_rs2_src), 
		.in_itype_immed(decode_itype_immed), 
		.in_mem_read(decode_mem_read), 
		.in_mem_write(decode_mem_write),
		.in_PC_next(decode_PC_next),
		.in_branch_type(decode_branch_type),
		.in_fwd_stall(forwarding_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_upper_immed(decode_upper_immed),
		.in_csr_address(decode_csr_address),
		.in_is_csr(decode_is_csr),
		.in_csr_mask(decode_csr_mask),
		.in_curr_PC(f_d_curr_PC),
		.in_jal(decode_jal),
		.in_jal_offset(decode_jal_offset),
		.in_freeze(total_freeze),
		.in_valid(decode_valid),

		.out_csr_address(d_e_csr_address),
		.out_is_csr(d_e_is_csr),
		.out_csr_mask(d_e_csr_mask),
		.out_rd(d_e_rd),
		.out_rs1(d_e_rs1),
		.out_rs2(d_e_rs2),
		.out_reg_data(d_e_reg_data),
		.out_alu_op(d_e_alu_op),
		.out_wb(d_e_wb),
		.out_rs2_src(d_e_rs2_src), 
		.out_itype_immed(d_e_itype_immed), 
		.out_mem_read(d_e_mem_read),
		.out_mem_write(d_e_mem_write),
		.out_branch_type(d_e_branch_type),
		.out_upper_immed(d_e_upper_immed),
		.out_curr_PC(d_e_curr_PC),
		.out_jal(d_e_jal),
		.out_jal_offset(d_e_jal_offset),
		.out_PC_next(d_e_PC_next),
		.out_valid(d_e_valid)
	);

VX_execute vx_execute(
		.in_rd(d_e_rd),
		.in_rs1(d_e_rs1),
		.in_rs2(d_e_rs2),
		.in_reg_data(d_e_reg_data),
		.in_alu_op(d_e_alu_op),
		.in_wb(d_e_wb),
		.in_rs2_src(d_e_rs2_src),
		.in_itype_immed(d_e_itype_immed),
		.in_mem_read(d_e_mem_read),
		.in_mem_write(d_e_mem_write),
		.in_PC_next(d_e_PC_next),
		.in_branch_type(d_e_branch_type),
		.in_upper_immed(d_e_upper_immed),
		.in_csr_address(d_e_csr_address),
		.in_is_csr(d_e_is_csr),
		.in_csr_data(csr_decode_csr_data),
		.in_csr_mask(d_e_csr_mask),
		.in_jal(d_e_jal),
		.in_jal_offset(d_e_jal_offset),
		.in_curr_PC(d_e_curr_PC),
		.in_valid(d_e_valid),

		.out_csr_address(execute_csr_address),
		.out_is_csr(execute_is_csr),
		.out_csr_result(execute_csr_result),
		.out_alu_result(execute_alu_result),
		.out_rd(execute_rd),
		.out_wb(execute_wb),
		.out_rs1(execute_rs1),
		.out_rs2(execute_rs2),
		.out_reg_data(execute_reg_data),
		.out_mem_read(execute_mem_read),
		.out_mem_write(execute_mem_write),
		.out_jal(execute_jal),
		.out_jal_dest(execute_jal_dest),
		.out_branch_offset(execute_branch_offset),
		.out_branch_stall(execute_branch_stall),
		.out_PC_next(execute_PC_next),
		.out_valid(execute_valid)
	);


VX_e_m_reg vx_e_m_reg(
		.clk(clk),
		.in_alu_result(execute_alu_result),
		.in_rd(execute_rd),
		.in_wb(execute_wb),
		.in_rs1(execute_rs1),
		.in_rs2(execute_rs2),
		.in_reg_data(execute_reg_data),
		.in_mem_read(execute_mem_read),
		.in_mem_write(execute_mem_write),
		.in_PC_next(execute_PC_next),
		.in_csr_address(execute_csr_address),
		.in_is_csr(execute_is_csr),
		.in_csr_result(execute_csr_result),
		.in_curr_PC(d_e_curr_PC),
		.in_branch_offset(execute_branch_offset),
		.in_branch_type(d_e_branch_type),
		.in_jal(execute_jal),
		.in_jal_dest(execute_jal_dest),
		.in_freeze(total_freeze),
		.in_valid(execute_valid),

		.out_csr_address(e_m_csr_address),
		.out_is_csr(e_m_is_csr),
		.out_csr_result(e_m_csr_result),
		.out_alu_result(e_m_alu_result),
		.out_rd(e_m_rd),
		.out_wb(e_m_wb),
		.out_rs1(e_m_rs1),
		.out_rs2(e_m_rs2),
		.out_reg_data(e_m_reg_data),
		.out_mem_read(e_m_mem_read),
		.out_mem_write(e_m_mem_write),
		.out_curr_PC(e_m_curr_PC),
		.out_branch_offset(e_m_branch_offset),
		.out_branch_type(e_m_branch_type),
		.out_jal(e_m_jal),
		.out_jal_dest(e_m_jal_dest),
		.out_PC_next(e_m_PC_next),
		.out_valid(e_m_valid)
	);

VX_memory vx_memory(
		.in_alu_result(e_m_alu_result),
		.in_mem_read(e_m_mem_read), 
		.in_mem_write(e_m_mem_write),
		.in_rd(e_m_rd),
		.in_wb(e_m_wb),
		.in_rs1(e_m_rs1),
		.in_rs2(e_m_rs2),
		.in_rd2(e_m_reg_data[1]),
		.in_PC_next(e_m_PC_next),
		.in_curr_PC(e_m_curr_PC),
		.in_branch_offset(e_m_branch_offset),
		.in_branch_type(e_m_branch_type), 
		.in_valid(e_m_valid),
		.in_cache_driver_out_data(in_cache_driver_out_data),

		.out_alu_result(memory_alu_result),
		.out_mem_result(memory_mem_result),
		.out_rd(memory_rd),
		.out_wb(memory_wb),
		.out_rs1(memory_rs1),
		.out_rs2(memory_rs2),
		.out_branch_dir(memory_branch_dir),
		.out_branch_dest(memory_branch_dest),
		.out_delay(memory_delay),
		.out_PC_next(memory_PC_next),
		.out_valid(memory_valid),
		.out_cache_driver_in_address(out_cache_driver_in_address),
		.out_cache_driver_in_mem_read(out_cache_driver_in_mem_read),
		.out_cache_driver_in_mem_write(out_cache_driver_in_mem_write),
		.out_cache_driver_in_data(out_cache_driver_in_data),
		.out_cache_driver_in_valid(out_cache_driver_in_valid)
	);

VX_m_w_reg vx_m_w_reg(
		.clk(clk),
		.in_alu_result(memory_alu_result),
		.in_mem_result(memory_mem_result),
		.in_rd(memory_rd),
		.in_wb(memory_wb),
		.in_rs1(memory_rs1),
		.in_rs2(memory_rs2),
		.in_PC_next(memory_PC_next),
		.in_freeze(total_freeze),
		.in_valid(memory_valid),

		.out_alu_result(m_w_alu_result),
		.out_mem_result(m_w_mem_result),
		.out_rd(m_w_rd),
		.out_wb(m_w_wb),
		.out_rs1(m_w_rs1),
		.out_rs2(m_w_rs2),
		.out_PC_next(m_w_PC_next),
		.out_valid(m_w_valid)
	);


VX_writeback vx_writeback(
		.in_alu_result(m_w_alu_result),
		.in_mem_result(m_w_mem_result),
		.in_rd(m_w_rd),
		.in_wb(m_w_wb),
		.in_PC_next(m_w_PC_next),

		.out_write_data(writeback_write_data),
		.out_rd(writeback_rd),
		.out_wb(writeback_wb)
	);


VX_forwarding vx_forwarding(
		.in_decode_src1(decode_rs1),
		.in_decode_src2(decode_rs2),
		.in_decode_csr_address(decode_csr_address), 

		.in_execute_dest(execute_rd),
		.in_execute_wb(execute_wb),
		.in_execute_alu_result(execute_alu_result),
		.in_execute_PC_next(execute_PC_next),
		.in_execute_is_csr(execute_is_csr),
		.in_execute_csr_address(execute_csr_address),

		.in_memory_dest(memory_rd),
		.in_memory_wb(memory_wb),
		.in_memory_alu_result(memory_alu_result),
		.in_memory_mem_data(memory_mem_result),
		.in_memory_PC_next(memory_PC_next),
		.in_memory_is_csr(e_m_is_csr),
		.in_memory_csr_address(e_m_csr_address),
		.in_memory_csr_result(e_m_csr_result),

		.in_writeback_dest(m_w_rd),
		.in_writeback_wb(m_w_wb),
		.in_writeback_alu_result(m_w_alu_result),
		.in_writeback_mem_data(m_w_mem_result),
		.in_writeback_PC_next(m_w_PC_next),

		.out_src1_fwd(forwarding_src1_fwd),
		.out_src2_fwd(forwarding_src2_fwd),
		.out_csr_fwd(forwarding_csr_fwd),
		.out_src1_fwd_data(forwarding_src1_fwd_data),
		.out_src2_fwd_data(forwarding_src2_fwd_data),
		.out_csr_fwd_data(forwarding_csr_fwd_data),
		.out_fwd_stall(forwarding_fwd_stall)
	);

VX_csr_handler vx_csr_handler(
		.clk(clk),
		.in_decode_csr_address(decode_csr_address),
		.in_mem_csr_address(e_m_csr_address),
		.in_mem_is_csr(e_m_is_csr),
		.in_mem_csr_result(e_m_csr_result),
		.in_wb_valid(m_w_valid),

		.out_decode_csr_data(csr_decode_csr_data)
	);




endmodule // Vortex





