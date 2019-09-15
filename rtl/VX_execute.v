
`include "VX_define.v"

module VX_execute (
		VX_frE_to_bckE_req_inter    VX_bckE_req,
		VX_forward_exe_inter        VX_fwd_exe,
		input wire[31:0]            in_csr_data,

		VX_mem_req_inter            VX_exe_mem_req,
		output wire[11:0]           out_csr_address,
		output wire                 out_is_csr,
		output reg[31:0]            out_csr_result,
		output wire                 out_jal,
		output wire[31:0]           out_jal_dest,
		output wire                 out_branch_stall
	);


		wire[`NT_M1:0][31:0] in_a_reg_data;
		wire[`NT_M1:0][31:0] in_b_reg_data;
		wire[4:0]            in_alu_op;
		wire                 in_rs2_src;
		wire[31:0]           in_itype_immed;
		wire[2:0]            in_branch_type;
		wire[19:0]           in_upper_immed;
		wire[31:0]           in_csr_mask;
		wire                 in_jal;
		wire[31:0]           in_jal_offset;
		wire[31:0]           in_curr_PC;

		assign in_a_reg_data  = VX_bckE_req.a_reg_data;
		assign in_b_reg_data  = VX_bckE_req.b_reg_data;
		assign in_alu_op      = VX_bckE_req.alu_op;
		assign in_rs2_src     = VX_bckE_req.rs2_src;
		assign in_itype_immed = VX_bckE_req.itype_immed;
		assign in_branch_type = VX_bckE_req.branch_type;
		assign in_upper_immed = VX_bckE_req.upper_immed;
		assign in_csr_mask    = VX_bckE_req.csr_mask;
		assign in_jal         = VX_bckE_req.jal;
		assign in_jal_offset  = VX_bckE_req.jal_offset;
		assign in_curr_PC     = VX_bckE_req.curr_PC;

		genvar index_out_reg;
		generate
			for (index_out_reg = 0; index_out_reg < `NT; index_out_reg = index_out_reg + 1)
				begin
					VX_alu vx_alu(
						// .in_reg_data   (in_reg_data[1:0]),
						.in_1          (in_a_reg_data[index_out_reg]),
						.in_2          (in_b_reg_data[index_out_reg]),
						.in_rs2_src    (in_rs2_src),
						.in_itype_immed(in_itype_immed),
						.in_upper_immed(in_upper_immed),
						.in_alu_op     (in_alu_op),
						.in_csr_data   (in_csr_data),
						.in_curr_PC    (in_curr_PC),
						.out_alu_result(VX_exe_mem_req.alu_result[index_out_reg])
					);
				end
		endgenerate


		assign out_jal_dest = $signed(in_a_reg_data[0]) + $signed(in_jal_offset);
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


		assign VX_exe_mem_req.mem_read      = VX_bckE_req.mem_read;
		assign VX_exe_mem_req.mem_write     = VX_bckE_req.mem_write;
		assign VX_exe_mem_req.wb            = VX_bckE_req.wb;
		assign VX_exe_mem_req.rs1           = VX_bckE_req.rs1;
		assign VX_exe_mem_req.rs2           = VX_bckE_req.rs2;
		assign VX_exe_mem_req.rd            = VX_bckE_req.rd;
		assign VX_exe_mem_req.rd2           = VX_bckE_req.b_reg_data;
		assign VX_exe_mem_req.wb            = VX_bckE_req.wb;
		assign VX_exe_mem_req.PC_next       = VX_bckE_req.PC_next;
		assign VX_exe_mem_req.curr_PC       = VX_bckE_req.curr_PC;
		assign VX_exe_mem_req.branch_offset = VX_bckE_req.itype_immed;
		assign VX_exe_mem_req.branch_type   = VX_bckE_req.branch_type;
		assign VX_exe_mem_req.valid         = VX_bckE_req.valid;
		assign VX_exe_mem_req.warp_num      = VX_bckE_req.warp_num;


		assign VX_fwd_exe.dest              = VX_exe_mem_req.rd;
		assign VX_fwd_exe.wb                = VX_exe_mem_req.wb;
		assign VX_fwd_exe.alu_result        = VX_exe_mem_req.alu_result;
		assign VX_fwd_exe.PC_next           = VX_exe_mem_req.PC_next;
		assign VX_fwd_exe.warp_num          = VX_exe_mem_req.warp_num;


		assign out_is_csr        = VX_bckE_req.is_csr;
		assign out_csr_address   = VX_bckE_req.csr_address;


endmodule // VX_execute
