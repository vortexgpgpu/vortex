
`include "VX_define.v"


module VX_writeback (
		input wire[31:0] in_alu_result,
		input wire[31:0] in_mem_result,
		input wire[4:0]  in_rd,
		input wire[1:0]  in_wb,
		input wire[31:0] in_PC_next,

		output wire[31:0] out_write_data,
		output wire[4:0] out_rd,
		output wire[1:0] out_wb
	);

		wire is_jal;
		wire uses_alu;


		assign is_jal   = in_wb == `WB_JAL;
		assign uses_alu = in_wb == `WB_ALU;

		assign out_write_data = is_jal ? in_PC_next :
										uses_alu ? in_alu_result :
													in_mem_result;


		assign out_rd = in_rd;
		assign out_wb = in_wb;


endmodule // VX_writeback