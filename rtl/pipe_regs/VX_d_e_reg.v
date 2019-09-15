

`include "VX_define.v"

module VX_d_e_reg (
		input wire               clk,
		input wire               reset,
		input wire               in_fwd_stall,
		input wire               in_branch_stall,
		input wire               in_freeze,
		input wire               in_gpr_stall,
		VX_frE_to_bckE_req_inter VX_frE_to_bckE_req,


		VX_frE_to_bckE_req_inter VX_bckE_req
	);


		wire stall = in_freeze;
		wire flush = (in_fwd_stall == `STALL) || (in_branch_stall == `STALL) || (in_gpr_stall == `STALL);


	VX_generic_register #(.N(489)) d_e_reg 
	(
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(flush),
		.in   ({VX_frE_to_bckE_req.csr_address, VX_frE_to_bckE_req.is_csr, VX_frE_to_bckE_req.csr_mask, VX_frE_to_bckE_req.rd, VX_frE_to_bckE_req.rs1, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.a_reg_data, VX_frE_to_bckE_req.b_reg_data, VX_frE_to_bckE_req.alu_op, VX_frE_to_bckE_req.wb, VX_frE_to_bckE_req.rs2_src, VX_frE_to_bckE_req.itype_immed, VX_frE_to_bckE_req.mem_read, VX_frE_to_bckE_req.mem_write, VX_frE_to_bckE_req.branch_type, VX_frE_to_bckE_req.upper_immed, VX_frE_to_bckE_req.curr_PC, VX_frE_to_bckE_req.jal, VX_frE_to_bckE_req.jal_offset, VX_frE_to_bckE_req.PC_next, VX_frE_to_bckE_req.valid, VX_frE_to_bckE_req.warp_num}),
		.out  ({VX_bckE_req.csr_address       , VX_bckE_req.is_csr       , VX_bckE_req.csr_mask       , VX_bckE_req.rd       , VX_bckE_req.rs1       , VX_bckE_req.rs2       , VX_bckE_req.a_reg_data       , VX_bckE_req.b_reg_data       , VX_bckE_req.alu_op       , VX_bckE_req.wb       , VX_bckE_req.rs2_src       , VX_bckE_req.itype_immed       , VX_bckE_req.mem_read       , VX_bckE_req.mem_write       , VX_bckE_req.branch_type       , VX_bckE_req.upper_immed       , VX_bckE_req.curr_PC       , VX_bckE_req.jal       , VX_bckE_req.jal_offset       , VX_bckE_req.PC_next       , VX_bckE_req.valid       , VX_bckE_req.warp_num})
	);


endmodule




