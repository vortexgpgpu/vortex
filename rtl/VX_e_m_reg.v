

`include "VX_define.v"


module VX_e_m_reg (
		input wire        clk,
		input wire[31:0]  in_alu_result,
		input wire[4:0]   in_rd,
		input wire[1:0]   in_wb,
		input wire[4:0]   in_rs1,
		input wire[31:0]  in_rd1,
		input wire[4:0]   in_rs2,
		input wire[31:0]  in_rd2,
		input wire[2:0]   in_mem_read, // NEW
		input wire[2:0]   in_mem_write, // NEW
		input wire[31:0]  in_PC_next,
		input wire[11:0]  in_csr_address,
		input wire        in_is_csr,
		input wire[31:0]  in_csr_result,
		input wire[31:0]  in_curr_PC,
		input wire[31:0]  in_branch_offset,
		input wire[2:0]   in_branch_type,
		input wire        in_jal,
		input wire[31:0]  in_jal_dest,
		input wire        in_freeze,
		input wire        in_valid,

		output wire[11:0] out_csr_address,
		output wire       out_is_csr,
		output wire[31:0] out_csr_result,
		output wire[31:0] out_alu_result,
		output wire[4:0]  out_rd,
		output wire[1:0]  out_wb,
		output wire[4:0]  out_rs1,
		output wire[31:0] out_rd1,
		output wire[31:0] out_rd2,
		output wire[4:0]  out_rs2,
		output wire[2:0]  out_mem_read,
		output wire[2:0]  out_mem_write,
		output wire[31:0] out_curr_PC,
		output wire[31:0] out_branch_offset,
		output wire[2:0]  out_branch_type,
		output wire       out_jal,
		output wire[31:0] out_jal_dest,
		output wire[31:0] out_PC_next,
		output wire       out_valid
	);


		reg[31:0] alu_result;
		reg[4:0]  rd;
		reg[4:0]  rs1;
		reg[31:0] rd1;
		reg[4:0]  rs2;
		reg[31:0] rd2;
		reg[1:0]  wb;
		reg[31:0] PC_next;
		reg[2:0]  mem_read;
		reg[2:0]  mem_write;
		reg[11:0] csr_address;
		reg       is_csr;
		reg[31:0] csr_result;
		reg[31:0] curr_PC;
		reg[31:0] branch_offset;
		reg[2:0]  branch_type;
		reg       jal;
		reg[31:0] jal_dest;
		reg       valid;


		initial begin
			alu_result    = 0;
			rd            = 0;
			rs1           = 0;
			rd1           = 0;
			rs2           = 0;
			rd2           = 0;
			wb            = 0;
			PC_next       = 0;
			mem_read      = `NO_MEM_READ;
			mem_write     = `NO_MEM_WRITE;
			csr_address   = 0;
			is_csr        = 0;
			csr_result    = 0;
			curr_PC       = 0;
			branch_offset = 0;
			branch_type   = 0;
			jal           = `NO_JUMP;
			jal_dest      = 0;
			valid         = 0;
		end



		assign out_alu_result    = alu_result;
		assign out_rd            = rd;
		assign out_rs1           = rs1;
		assign out_rs2           = rs2;
		assign out_wb            = wb;
		assign out_PC_next       = PC_next;
		assign out_mem_read      = mem_read;
		assign out_mem_write     = mem_write;
		assign out_rd1           = rd1;
		assign out_rd2           = rd2;
		assign out_csr_address   = csr_address;
		assign out_is_csr        = is_csr;
		assign out_csr_result    = csr_result;
		assign out_curr_PC       = curr_PC;
		assign out_branch_offset = branch_offset;
		assign out_branch_type   = branch_type;
		assign out_jal           = jal;
		assign out_jal_dest      = jal_dest;
		assign out_valid         = valid;
		

		always @(posedge clk) begin
			if(in_freeze == 1'b0) begin
				alu_result    <= in_alu_result;
				rd            <= in_rd;
				rs1           <= in_rs1;
				rs2           <= in_rs2;
				wb            <= in_wb;
				PC_next       <= in_PC_next;
				mem_read      <= in_mem_read;
				mem_write     <= in_mem_write;
				rd1           <= in_rd1;
				rd2           <= in_rd2;
				csr_address   <= in_csr_address;
				is_csr        <= in_is_csr;
				csr_result    <= in_csr_result;
				curr_PC       <= in_curr_PC;
				branch_offset <= in_branch_offset;
				branch_type   <= in_branch_type;
				jal           <= in_jal;
				jal_dest      <= in_jal_dest;
				valid         <= in_valid;
			end
		end

endmodule // VX_e_m_reg





