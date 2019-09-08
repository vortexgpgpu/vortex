

`include "VX_define.v"

module VX_m_w_reg (
		input wire           clk,
		input wire           reset,
		input wire           in_freeze,

		VX_inst_mem_wb_inter VX_mem_wb,
		VX_mw_wb_inter       VX_mw_wb
	);

		wire flush = 0;
		wire stall = in_freeze;


		VX_generic_register #(.N(303)) m_w_reg 
		(
			.clk  (clk),
			.reset(reset),
			.stall(stall),
			.flush(flush),
			.in   ({VX_mem_wb.alu_result, VX_mem_wb.mem_result, VX_mem_wb.rd, VX_mem_wb.wb, VX_mem_wb.PC_next, VX_mem_wb.valid, VX_mem_wb.warp_num}),
			.out  ({VX_mw_wb.alu_result , VX_mw_wb.mem_result , VX_mw_wb.rd , VX_mw_wb.wb , VX_mw_wb.PC_next , VX_mw_wb.valid , VX_mw_wb.warp_num })
		);



endmodule // VX_m_w_reg



