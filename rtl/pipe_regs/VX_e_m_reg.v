

`include "VX_define.v"


module VX_e_m_reg (
		input wire        clk,
		input wire        reset,
		input wire        in_freeze,
		input wire[11:0]  in_csr_address,
		input wire        in_is_csr,
		input wire[31:0]  in_csr_result,
		input wire        in_jal,
		input wire[31:0]  in_jal_dest,
		VX_mem_req_inter  VX_exe_mem_req,


		VX_mem_req_inter  VX_mem_req,
		output wire[11:0] out_csr_address,
		output wire       out_is_csr,
		output wire[31:0] out_csr_result,
		output wire       out_jal,
		output wire[31:0] out_jal_dest
	);


		wire flush = 0;
		wire stall = in_freeze;

	VX_generic_register #(.N(464)) f_d_reg 
	(
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(flush),
		.in   ({in_csr_address , in_is_csr , in_csr_result , in_jal , in_jal_dest , VX_exe_mem_req.alu_result, VX_exe_mem_req.mem_read, VX_exe_mem_req.mem_write, VX_exe_mem_req.rd, VX_exe_mem_req.wb, VX_exe_mem_req.rs1, VX_exe_mem_req.rs2, VX_exe_mem_req.rd2, VX_exe_mem_req.PC_next, VX_exe_mem_req.curr_PC, VX_exe_mem_req.branch_offset, VX_exe_mem_req.branch_type, VX_exe_mem_req.valid, VX_exe_mem_req.warp_num}),
		.out  ({out_csr_address, out_is_csr, out_csr_result, out_jal, out_jal_dest, VX_mem_req.alu_result    , VX_mem_req.mem_read    , VX_mem_req.mem_write    , VX_mem_req.rd    , VX_mem_req.wb    , VX_mem_req.rs1    , VX_mem_req.rs2    , VX_mem_req.rd2    , VX_mem_req.PC_next    , VX_mem_req.curr_PC    , VX_mem_req.branch_offset    , VX_mem_req.branch_type    , VX_mem_req.valid    , VX_mem_req.warp_num})
	);

endmodule // VX_e_m_reg





