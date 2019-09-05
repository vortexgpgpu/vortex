

`include "VX_define.v"

module VX_m_w_reg (
		input wire           clk,
		input wire           reset,
		VX_inst_mem_wb_inter VX_mem_wb,

		input wire           in_freeze,

		output wire[`NT_M1:0][31:0] out_alu_result,
		output wire[`NT_M1:0][31:0] out_mem_result, // NEW
		output wire[4:0]  out_rd,
		output wire[1:0]  out_wb,
		output wire[4:0]  out_rs1,
		output wire[4:0]  out_rs2,
		output wire[31:0] out_PC_next,
		output wire[`NT_M1:0] out_valid,
	    output wire[`NW_M1:0] out_warp_num
	);

		wire flush = 0;
		wire stall = in_freeze;


		VX_generic_register #(.N(313)) m_w_reg 
		(
			.clk  (clk),
			.reset(reset),
			.stall(stall),
			.flush(flush),
			.in   ({VX_mem_wb.alu_result, VX_mem_wb.mem_result, VX_mem_wb.rd, VX_mem_wb.wb, VX_mem_wb.rs1, VX_mem_wb.rs2, VX_mem_wb.PC_next, VX_mem_wb.valid, VX_mem_wb.warp_num}),
			.out  ({out_alu_result      , out_mem_result      , out_rd      , out_wb      , out_rs1      , out_rs2      , out_PC_next      , out_valid      , out_warp_num      })
		);



endmodule // VX_m_w_reg



