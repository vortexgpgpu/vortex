
`include "VX_define.v"

module VX_execute (
		input wire[4:0]   in_rd,
		input wire[4:0]   in_rs1,
		input wire[31:0]  in_rd1,
		input wire[4:0]   in_rs2,
		input wire[31:0]  in_rd2,
		input wire[3:0]   in_alu_op,
		input wire[1:0]   in_wb,
		input wire        in_rs2_src, // NEW
		input wire[31:0]  in_itype_immed, // new
		input wire[2:0]   in_mem_read, // NEW
		input wire[2:0]   in_mem_write, // NEW
		input wire[31:0]  in_PC_next,
		input wire[2:0]   in_branch_type,
		input wire[19:0]  in_upper_immed,
		input wire[11:0]  in_csr_address, // done
		input wire        in_is_csr, // done
		input wire[31:0]  in_csr_data, // done
		input wire[31:0]  in_csr_mask, // done
		input wire        in_jal,
		input wire[31:0]  in_jal_offset,
		input wire[31:0]  in_curr_PC,
		input wire        in_valid,

		output wire[11:0] out_csr_address,
		output wire       out_is_csr,
		output reg[31:0]  out_csr_result,
		output reg[31:0]  out_alu_result,
		output wire[4:0]  out_rd,
		output wire[1:0]  out_wb,
		output wire[4:0]  out_rs1,
		output wire[31:0] out_rd1,
		output wire[4:0]  out_rs2,
		output wire[31:0] out_rd2,
		output wire[2:0]  out_mem_read,
		output wire[2:0]  out_mem_write,
		output wire       out_jal,
		output wire[31:0] out_jal_dest,
		output wire[31:0] out_branch_offset,
		output wire       out_branch_stall,
		output wire[31:0] out_PC_next,
		output wire       out_valid
	);

		wire which_in2;

		wire[31:0] ALU_in1;
		wire[31:0] ALU_in2;
		wire[31:0] upper_immed;


		assign which_in2  = in_rs2_src == `RS2_IMMED;

		assign ALU_in1 = in_rd1;

		assign ALU_in2 = which_in2 ? in_itype_immed : in_rd2;


		assign upper_immed = {in_upper_immed, {12{1'b0}}};
		assign out_jal_dest = $signed(in_rd1) + $signed(in_jal_offset);
		assign out_jal      = in_jal;


		// always @(*) begin
		// 	$display("EXECUTE CURR_PC: %h",in_curr_PC);
		// end

		always @(*) begin
			case(in_alu_op)
				`ADD:
					begin
						out_alu_result = $signed(ALU_in1) + $signed(ALU_in2);
						out_csr_result = 32'hdeadbeef;
					end
				`SUB:
					begin
						out_alu_result = $signed(ALU_in1) - $signed(ALU_in2);
						// $display("PC: %h ----> %h and %h",in_curr_PC, $signed(ALU_in1), $signed(ALU_in2));
						out_csr_result = 32'hdeadbeef;
					end
				`SLLA:
					begin
						out_alu_result = ALU_in1 << ALU_in2[4:0];
						out_csr_result = 32'hdeadbeef;
					end
				`SLT:
					begin
						out_alu_result = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
						out_csr_result = 32'hdeadbeef;
					end
				`SLTU:
					begin

						out_alu_result = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
						out_csr_result = 32'hdeadbeef;
					end
				`XOR:
					begin
						out_alu_result = ALU_in1 ^ ALU_in2;
						out_csr_result = 32'hdeadbeef;
					end
				`SRL:
					begin
						out_alu_result = ALU_in1 >> ALU_in2[4:0];
						out_csr_result = 32'hdeadbeef;
					end
				`SRA:
					begin
						out_alu_result    = $signed(ALU_in1)  >>> ALU_in2[4:0];
						// $display("Shifting right arithmatic - PC: %h\t%h >>> %h = %h",in_curr_PC, $signed(ALU_in1), ALU_in2, out_alu_result);
						out_csr_result    = 32'hdeadbeef;
					end
				`OR:
					begin
						out_alu_result = ALU_in1 | ALU_in2;
						out_csr_result = 32'hdeadbeef;
					end
				`AND:
					begin
						out_alu_result = ALU_in2 & ALU_in1;
						out_csr_result = 32'hdeadbeef;
					end
				`SUBU:
					begin
						if (ALU_in1 >= ALU_in2) begin
							out_alu_result = 32'h0;
						end else begin
							out_alu_result = 32'hffffffff;

						end
						out_csr_result = 32'hdeadbeef;
					end
				`LUI_ALU:
					begin
						out_alu_result = upper_immed;
						out_csr_result = 32'hdeadbeef;
					end
				`AUIPC_ALU:
					begin
						out_alu_result = $signed(in_curr_PC) + $signed(upper_immed);
						out_csr_result = 32'hdeadbeef;
					end
				`CSR_ALU_RW:
					begin
						out_alu_result = in_csr_data;
						out_csr_result = in_csr_mask;
					end
				`CSR_ALU_RS:
					begin
						out_alu_result = in_csr_data;
						out_csr_result = in_csr_data | in_csr_mask;
					end
				`CSR_ALU_RC:
					begin
						out_alu_result = in_csr_data;
						out_csr_result = in_csr_data & (32'hFFFFFFFF - in_csr_mask);
					end
				default:
					begin
						out_alu_result = 32'h0;
						out_csr_result = 32'hdeadbeef;
					end
			endcase // in_alu_op
		end


		assign out_branch_stall = ((in_branch_type != `NO_BRANCH) || in_jal ) ? `STALL : `NO_STALL;



		assign out_rd            = in_rd;
		assign out_wb            = in_wb;
		assign out_mem_read      = in_mem_read;
		assign out_mem_write     = in_mem_write;
		assign out_rs1           = in_rs1;
		assign out_rd1           = in_rd1;
		assign out_rd2           = in_rd2;
		assign out_rs2           = in_rs2;
		assign out_PC_next       = in_PC_next;
		assign out_is_csr        = in_is_csr;
		assign out_csr_address   = in_csr_address;
		assign out_branch_offset = in_itype_immed;
		assign out_valid         = in_valid;


endmodule // VX_execute
