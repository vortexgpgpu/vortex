
`include "VX_define.v"


module VX_writeback (
		/* verilator lint_off UNUSED */
		input wire       clk,
		/* verilator lint_off UNUSED */
		input wire[31:0] in_alu_result[`NT_M1:0],
		input wire[31:0] in_mem_result[`NT_M1:0],
		input wire[4:0]  in_rd,
		input wire[1:0]  in_wb,
		input wire[31:0] in_PC_next,
		/* verilator lint_off UNUSED */
		input wire       in_valid[`NT_M1:0],
		/* verilator lint_on UNUSED */
		input wire [`NW_M1:0]  in_warp_num,

		output wire[31:0] out_write_data[`NT_M1:0],
		output wire[4:0] out_rd,
		output wire[1:0] out_wb,
		output wire[`NW_M1:0]  out_warp_num
	);

		wire is_jal;
		wire uses_alu;

		wire[31:0] out_pc_data[`NT_M1:0];


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

		assign out_write_data = is_jal ? out_pc_data :
										uses_alu ? in_alu_result :
													in_mem_result;


		// always @(negedge clk) begin
		// 	if (in_wb != 0)  begin
		// 		$display("[%h] WB Data: %h {%h}, to register: %d [%d %d]",in_PC_next - 4, out_write_data[0], in_mem_result[0], in_rd, in_valid[0], in_valid[1]);
		// 	end
		// end

		assign out_rd = in_rd;
		assign out_wb = in_wb;
		assign out_warp_num = in_warp_num;


endmodule // VX_writeback