

`include "VX_define.v"

module VX_d_e_reg (
		input wire           clk,
		input wire[4:0]      in_rd,
		input wire[4:0]      in_rs1,
		input wire[4:0]      in_rs2,
		input wire[31:0]     in_a_reg_data[`NT_M1:0],
		input wire[31:0]     in_b_reg_data[`NT_M1:0],
		input wire[4:0]      in_alu_op,
		input wire[1:0]      in_wb,
		input wire           in_rs2_src, // NEW
		input wire[31:0]     in_itype_immed, // new
		input wire[2:0]      in_mem_read, // NEW
		input wire[2:0]      in_mem_write,
		input wire[31:0]     in_PC_next,
		input wire[2:0]      in_branch_type,
		input wire           in_fwd_stall,
		input wire           in_branch_stall,
		input wire[19:0]     in_upper_immed,
		input wire[11:0]     in_csr_address, // done
		input wire           in_is_csr, // done
		input wire[31:0]     in_csr_mask, // done
		input wire[31:0]     in_curr_PC,
		input wire           in_jal,
		input wire[31:0]     in_jal_offset,
		input wire           in_freeze,
		input wire           in_clone_stall,
		input wire           in_valid[`NT_M1:0],
		input wire[`NW_M1:0] in_warp_num,

		output wire[11:0]     out_csr_address, // done
		output wire           out_is_csr, // done
		output wire[31:0]     out_csr_mask, // done
		output wire[4:0]      out_rd,
		output wire[4:0]      out_rs1,
		output wire[4:0]      out_rs2,
		output wire[31:0]     out_a_reg_data[`NT_M1:0],
		output wire[31:0]     out_b_reg_data[`NT_M1:0],
		output wire[4:0]      out_alu_op,
		output wire[1:0]      out_wb,
		output wire           out_rs2_src, // NEW
		output wire[31:0]     out_itype_immed, // new
		output wire[2:0]      out_mem_read,
		output wire[2:0]      out_mem_write,
		output wire[2:0]      out_branch_type,
		output wire[19:0]     out_upper_immed,
		output wire[31:0]     out_curr_PC,
		output wire           out_jal,
		output wire[31:0]     out_jal_offset,
		output wire[31:0]     out_PC_next,
		output wire           out_valid[`NT_M1:0],
	    output wire[`NW_M1:0] out_warp_num
	);


		reg[4:0]  rd;
		reg[4:0]  rs1;
		reg[4:0]  rs2;
		reg[31:0] a_reg_data[`NT_M1:0];
		reg[31:0] b_reg_data[`NT_M1:0];
		reg[4:0]  alu_op;
		reg[1:0]  wb;
		reg[31:0] PC_next_out;
		reg       rs2_src;
		reg[31:0] itype_immed;
		reg[2:0]  mem_read;
		reg[2:0]  mem_write;
		reg[2:0]  branch_type;
		reg[19:0] upper_immed;
		reg[11:0] csr_address;
		reg       is_csr;
		reg[31:0] csr_mask;
		reg[31:0] curr_PC;
		reg       jal;
		reg[31:0] jal_offset;
		reg       valid[`NT_M1:0];

		reg[31:0] reg_data_z[`NT_M1:0];
		reg       valid_z[`NT_M1:0];

		reg[`NW_M1:0] warp_num;

		integer ini_reg;
		initial begin
			rd          = 0;
			rs1         = 0;
			for (ini_reg = 0; ini_reg < `NT; ini_reg = ini_reg + 1)
			begin
				a_reg_data[ini_reg]   = 0;
				b_reg_data[ini_reg]   = 0;
				reg_data_z[ini_reg]   = 0;
				valid[ini_reg]        = 0;
				valid_z[ini_reg]      = 0;
			end
			rs2         = 0;
			alu_op      = 0;
			wb          = `NO_WB;
			PC_next_out = 0;
			rs2_src     = 0;
			itype_immed = 0;
			mem_read    = `NO_MEM_READ;
			mem_write   = `NO_MEM_WRITE;
			branch_type = `NO_BRANCH;
			upper_immed = 0;
			csr_address = 0;
			is_csr      = 0;
			csr_mask    = 0;
			curr_PC     = 0;
			jal         = `NO_JUMP;
			jal_offset  = 0;
			warp_num    = 0;
		end

		wire stalling;

		assign stalling = (in_fwd_stall == `STALL) || (in_branch_stall == `STALL) || (in_clone_stall == `STALL);

		assign out_rd          = rd;
		assign out_rs1         = rs1;
		assign out_rs2         = rs2;
		assign out_a_reg_data  = a_reg_data;
		assign out_b_reg_data  = b_reg_data;
		assign out_alu_op      = alu_op;
		assign out_wb          = wb;
		assign out_PC_next     = PC_next_out;
		assign out_rs2_src     = rs2_src;
		assign out_itype_immed = itype_immed;
		assign out_mem_read    = mem_read;
		assign out_mem_write   = mem_write;
		assign out_branch_type = branch_type;
		assign out_upper_immed = upper_immed;
		assign out_csr_address = csr_address;
		assign out_is_csr      = is_csr;
		assign out_csr_mask    = csr_mask;
		assign out_jal         = jal;
		assign out_jal_offset  = jal_offset;
		assign out_curr_PC     = curr_PC;
		assign out_valid       = valid;
		assign out_warp_num    = warp_num;


		always @(posedge clk) begin
			if (in_freeze == 1'h0) begin
				rd          <= stalling ? 5'h0         : in_rd;
				rs1         <= stalling ? 5'h0         : in_rs1;
				rs2         <= stalling ? 5'h0         : in_rs2;
				a_reg_data  <= stalling ? reg_data_z   : in_a_reg_data;
				b_reg_data  <= stalling ? reg_data_z   : in_b_reg_data;
				alu_op      <= stalling ? `NO_ALU      : in_alu_op;
				wb          <= stalling ? `NO_WB       : in_wb;
				PC_next_out <= stalling ? 32'h0        : in_PC_next;
				rs2_src     <= stalling ? `RS2_REG     : in_rs2_src;
				itype_immed <= stalling ? 32'hdeadbeef : in_itype_immed;
				mem_read    <= stalling ? `NO_MEM_READ : in_mem_read;
				mem_write   <= stalling ? `NO_MEM_WRITE: in_mem_write;
				branch_type <= stalling ? `NO_BRANCH   : in_branch_type;
				upper_immed <= stalling ? 20'h0        : in_upper_immed;
				csr_address <= stalling ? 12'h0        : in_csr_address;
				is_csr      <= stalling ? 1'h0         : in_is_csr;
				csr_mask    <= stalling ? 32'h0        : in_csr_mask;
				jal         <= stalling ? `NO_JUMP     : in_jal;
				jal_offset  <= stalling ? 32'h0        : in_jal_offset;
				curr_PC     <= stalling ? 32'h0        : in_curr_PC;
				valid       <= stalling ? valid_z      : in_valid;
				warp_num    <= stalling ? 0            : in_warp_num;
			end
		end

endmodule




