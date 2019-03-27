
`include "VX_define.v"

module VX_execute (
		input wire[4:0]      in_rd,
		input wire[4:0]      in_rs1,
		input wire[4:0]      in_rs2,
		input wire[31:0]     in_reg_data[`NT_T2_M1:0],
		input wire[4:0]      in_alu_op,
		input wire[1:0]      in_wb,
		input wire           in_rs2_src, // NEW
		input wire[31:0]     in_itype_immed, // new
		input wire[2:0]      in_mem_read, // NEW
		input wire[2:0]      in_mem_write, // NEW
		input wire[31:0]     in_PC_next,
		input wire[2:0]      in_branch_type,
		input wire[19:0]     in_upper_immed,
		input wire[11:0]     in_csr_address, // done
		input wire           in_is_csr, // done
		input wire[31:0]     in_csr_data, // done
		input wire[31:0]     in_csr_mask, // done
		input wire           in_jal,
		input wire[31:0]     in_jal_offset,
		input wire[31:0]     in_curr_PC,
		input wire[`NT_M1:0] in_valid,

		output wire[11:0]     out_csr_address,
		output wire           out_is_csr,
		output reg[31:0]      out_csr_result,
		output reg[31:0]      out_alu_result[`NT_M1:0],
		output wire[4:0]      out_rd,
		output wire[1:0]      out_wb,
		output wire[4:0]      out_rs1,
		output wire[4:0]      out_rs2,
		output wire[31:0]     out_reg_data[`NT_T2_M1:0],
		output wire[2:0]      out_mem_read,
		output wire[2:0]      out_mem_write,
		output wire           out_jal,
		output wire[31:0]     out_jal_dest,
		output wire[31:0]     out_branch_offset,
		output wire           out_branch_stall,
		output wire[31:0]     out_PC_next,
		output wire[`NT_M1:0] out_valid
	);



	// VX_alu vx_alu(
	// 		.in_reg_data   (in_reg_data),
	// 		.in_rs2_src    (in_rs2_src),
	// 		.in_itype_immed(in_itype_immed),
	// 		.in_upper_immed(in_upper_immed),
	// 		.in_alu_op     (in_alu_op),
	// 		.in_csr_data   (in_csr_data),
	// 		.in_curr_PC    (in_curr_PC),
	// 		.out_alu_result(out_alu_result)
	// 	);

		genvar index;
		genvar index_2;
		generate  
		for (index=0; index <= `NT; index=index+2)  
		  begin: gen_code_label  
		  	assign index_2 = index / 2;
			VX_alu vx_alu(
				.in_reg_data   (in_reg_data[index+1:index]),
				.in_rs2_src    (in_rs2_src),
				.in_itype_immed(in_itype_immed),
				.in_upper_immed(in_upper_immed),
				.in_alu_op     (in_alu_op),
				.in_csr_data   (in_csr_data),
				.in_curr_PC    (in_curr_PC),
				.out_alu_result(out_alu_result[index_2])
			);
		  end  
		endgenerate  


		assign out_jal_dest = $signed(in_reg_data[0]) + $signed(in_jal_offset);
		assign out_jal      = in_jal;

		always @(*) begin

			case(in_alu_op)
				`CSR_ALU_RW: out_csr_result = in_csr_mask;
				`CSR_ALU_RS: out_csr_result = in_csr_data | in_csr_mask;
				`CSR_ALU_RC: out_csr_result = in_csr_data & (32'hFFFFFFFF - in_csr_mask);
				default:     out_csr_result = 32'hdeadbeef;
			endcase
		
		end




		assign out_branch_stall = ((in_branch_type != `NO_BRANCH) || in_jal ) ? `STALL : `NO_STALL;



		assign out_rd            = in_rd;
		assign out_wb            = in_wb;
		assign out_mem_read      = in_mem_read;
		assign out_mem_write     = in_mem_write;
		assign out_rs1           = in_rs1;
		assign out_reg_data      = in_reg_data;
		assign out_rs2           = in_rs2;
		assign out_PC_next       = in_PC_next;
		assign out_is_csr        = in_is_csr;
		assign out_csr_address   = in_csr_address;
		assign out_branch_offset = in_itype_immed;
		assign out_valid         = in_valid;


endmodule // VX_execute
