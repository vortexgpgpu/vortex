

`include "VX_define.v"

module VX_m_w_reg (
		input wire        clk,
		input wire[31:0]  in_alu_result[`NT_M1:0],
		input wire[31:0]  in_mem_result[`NT_M1:0], // NEW
		input wire[4:0]   in_rd,
		input wire[1:0]   in_wb,
		input wire[4:0]   in_rs1,
		input wire[4:0]   in_rs2,
		input wire[31:0]  in_PC_next,
		input wire        in_freeze,
		input wire        in_valid[`NT_M1:0],
		input wire[`NW_M1:0] in_warp_num,

		output wire[31:0] out_alu_result[`NT_M1:0],
		output wire[31:0] out_mem_result[`NT_M1:0], // NEW
		output wire[4:0]  out_rd,
		output wire[1:0]  out_wb,
		output wire[4:0]  out_rs1,
		output wire[4:0]  out_rs2,
		output wire[31:0] out_PC_next,
		output wire       out_valid[`NT_M1:0],
	    output wire[`NW_M1:0] out_warp_num
	);



		reg[31:0] alu_result[`NT_M1:0];
		reg[31:0] mem_result[`NT_M1:0];
		reg[4:0]  rd;
		reg[4:0]  rs1;
		reg[4:0]  rs2;
		reg[1:0]  wb;
		reg[31:0] PC_next;
		reg       valid[`NT_M1:0];
		reg[`NW_M1:0] warp_num;

		initial begin
			// alu_result  = 0;
			// mem_result  = 0;
			rd          = 0;
			rs1         = 0;
			rs2         = 0;
			wb          = 0;
			PC_next     = 0;
			warp_num    = 0;
			// valid       = 0;
		end

		assign out_alu_result  = alu_result;
		assign out_mem_result  = mem_result;
		assign out_rd          = rd;
		assign out_rs1         = rs1;
		assign out_rs2         = rs2;
		assign out_wb          = wb;
		assign out_PC_next     = PC_next;
		assign out_valid       = valid;
		assign out_warp_num    = warp_num;

		always @(posedge clk) begin
			if(in_freeze == 1'b0) begin
				alu_result   <= in_alu_result;
				mem_result   <= in_mem_result;
				rd           <= in_rd;
				rs1          <= in_rs1;
				rs2          <= in_rs2;
				wb           <= in_wb;
				PC_next      <= in_PC_next;
				valid        <= in_valid;
				warp_num     <= in_warp_num;
			end
		end



endmodule // VX_m_w_reg



