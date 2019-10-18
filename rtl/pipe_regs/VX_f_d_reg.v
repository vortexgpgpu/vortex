`include "VX_define.v"

module VX_f_d_reg (
	input wire             clk,
	input wire             reset,
	input wire             in_freeze,

	VX_inst_meta_inter     fe_inst_meta_fd,
	VX_inst_meta_inter     fd_inst_meta_de

);

	wire flush = 1'b0;
	wire stall = in_freeze == 1'b1;



	VX_generic_register #(.N(71)) f_d_reg 
	(
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(flush),
		.in   ({fe_inst_meta_fd.instruction, fe_inst_meta_fd.inst_pc, fe_inst_meta_fd.warp_num, fe_inst_meta_fd.valid}),
		.out  ({fd_inst_meta_de.instruction, fd_inst_meta_de.inst_pc, fd_inst_meta_de.warp_num, fd_inst_meta_de.valid})
	);


endmodule