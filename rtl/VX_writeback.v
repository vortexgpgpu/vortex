
`include "VX_define.v"


module VX_writeback (
		/* verilator lint_off UNUSED */
		input wire       clk,
		/* verilator lint_off UNUSED */
		input wire[`NT_M1:0][31:0] in_alu_result,
		input wire[`NT_M1:0][31:0] in_mem_result,
		input wire[4:0]  in_rd,
		input wire[1:0]  in_wb,
		input wire[31:0] in_PC_next,
		/* verilator lint_off UNUSED */
		input wire[`NT_M1:0]       in_valid,
		/* verilator lint_on UNUSED */
		input wire [`NW_M1:0]  in_warp_num,


		VX_wb_inter       VX_writeback_inter
	);

		wire is_jal;
		wire uses_alu;

		wire[`NT_M1:0][31:0] out_pc_data;


		// genvar index;
		// for (index=0; index < `NT; index=index+1)  
		//   assign out_pc_data[index] = in_PC_next;
		// generate 
		// endgenerate

		genvar i;
		generate
			for (i = 0; i < `NT; i=i+1)
			begin
				assign out_pc_data[i] = in_PC_next;
			end
		endgenerate

		// assign out_pc_data[0] = in_PC_next;

		// assign out_pc_data[1] = in_PC_next;

		assign is_jal   = in_wb == `WB_JAL;
		assign uses_alu = in_wb == `WB_ALU;

		assign VX_writeback_inter.write_data = is_jal ? out_pc_data :
										uses_alu ? in_alu_result :
													in_mem_result;

		assign VX_writeback_inter.wb_valid    = in_valid;
		assign VX_writeback_inter.rd          = in_rd;
		assign VX_writeback_inter.wb          = in_wb;
		assign VX_writeback_inter.wb_warp_num = in_warp_num;


endmodule // VX_writeback