
`include "VX_define.v"

module VX_decode(
	// Fetch Inputs
	input wire        clk,
	input wire[31:0]  in_instruction,
	input wire[31:0]  in_curr_PC,
	input wire        in_valid,
	// WriteBack inputs
	input wire[31:0]  in_write_data,
	input wire[4:0]   in_rd,
	input wire[1:0]   in_wb,

	// FORWARDING INPUTS
	input wire        in_src1_fwd,
	input wire[31:0]  in_src1_fwd_data,
	input wire        in_src2_fwd,
	input wire[31:0]  in_src2_fwd_data,

	output wire[11:0] out_csr_address, // done
	output wire       out_is_csr, // done
	output wire[31:0] out_csr_mask, // done

	// Outputs
	output wire[4:0]  out_rd,
	output wire[4:0]  out_rs1,
	output wire[31:0] out_rd1,
	output wire[4:0]  out_rs2,
	output wire[31:0] out_rd2,
	output wire[1:0]  out_wb,
	output wire[4:0]  out_alu_op,
	output wire       out_rs2_src, // NEW
	output reg[31:0]  out_itype_immed, // new
	output wire[2:0]  out_mem_read, // NEW
	output wire[2:0]  out_mem_write, // NEW
	output reg[2:0]   out_branch_type,
	output reg        out_branch_stall,
	output reg        out_jal,
	output reg[31:0]  out_jal_offset,
	output reg[19:0]  out_upper_immed,
	output wire[31:0] out_PC_next,
	output wire       out_valid
);

		wire[6:0]  curr_opcode;


		wire[31:0] rd1_register;
		wire[31:0] rd2_register;

		wire       is_itype;
		wire       is_rtype;
		wire       is_stype;
		wire       is_btype;
		wire       is_linst;
		wire       is_jal;
		wire       is_jalr;
		wire       is_lui;
		wire       is_auipc;
		wire       is_csr;
		wire       is_csr_immed;
		wire       is_e_inst;

		wire       write_register;

		wire[2:0]  func3;
		wire[6:0]  func7;
		wire[11:0] u_12;


		wire[7:0]  jal_b_19_to_12;
		wire       jal_b_11;
		wire[9:0]  jal_b_10_to_1;
		wire       jal_b_20;
		wire       jal_b_0;
		wire[20:0] jal_unsigned_offset;
		wire[31:0] jal_1_offset;

		wire[11:0] jalr_immed;
		wire[31:0] jal_2_offset;

		wire       jal_sys_cond1;
		wire       jal_sys_cond2;
		wire       jal_sys_jal;
		wire[31:0] jal_sys_off;

		wire       csr_cond1;
		wire       csr_cond2;

		wire[11:0] alu_tempp;
		wire       alu_shift_i;
		wire[11:0] alu_shift_i_immed;

		wire[1:0]  csr_type;

		reg[4:0]   csr_alu;
		reg[4:0]   alu_op;
		reg[4:0]   mul_alu;

		// always @(posedge clk) begin
		// 	$display("Decode: curr_pc: %h", in_curr_PC);
		// end

		VX_register_file vx_register_file(
			.clk(clk),
			.in_write_register(write_register),
			.in_rd(in_rd),
			.in_data(in_write_data),
			.in_src1(out_rs1),
			.in_src2(out_rs2),
			.out_src1_data(rd1_register),
			.out_src2_data(rd2_register)
		);

		assign out_valid = in_valid;

		assign write_register = (in_wb != 2'h0) ? (1'b1) : (1'b0);
		assign curr_opcode    = in_instruction[6:0];


		assign out_rd   = in_instruction[11:7];
		assign out_rs1  = in_instruction[19:15];
		assign out_rs2  = in_instruction[24:20];
		assign func3    = in_instruction[14:12];
		assign func7    = in_instruction[31:25];
		assign u_12     = in_instruction[31:20];


		assign out_PC_next = in_curr_PC + 32'h4;


		// Write Back sigal
		assign is_rtype     = (curr_opcode == `R_INST);
		assign is_linst     = (curr_opcode == `L_INST);
		assign is_itype     = (curr_opcode == `ALU_INST) || is_linst; 
		assign is_stype     = (curr_opcode == `S_INST);
		assign is_btype     = (curr_opcode == `B_INST);
		assign is_jal       = (curr_opcode == `JAL_INST);
		assign is_jalr      = (curr_opcode == `JALR_INST);
		assign is_lui       = (curr_opcode == `LUI_INST);
		assign is_auipc     = (curr_opcode == `AUIPC_INST);
		assign is_csr       = (curr_opcode == `SYS_INST) && (func3 != 0);
		assign is_csr_immed = (is_csr) && (func3[2] == 1);
		assign is_e_inst    = (curr_opcode == `SYS_INST) && (func3 == 0);


		// ch_print("DECODE: PC: {0}, INSTRUCTION: {1}", in_curr_PC, in_instruction);


		assign out_rd1 = ((is_jal == 1'b1) ? in_curr_PC : ((in_src1_fwd == 1'b1) ? in_src1_fwd_data : rd1_register));

		// always @(negedge clk) begin
		// 	if (in_curr_PC == 32'h800001f0) begin
		// 		$display("IN DECODE: Going to write to: %d with val: %h [%h, %h, %h]", out_rd, out_rd1, in_curr_PC, in_src1_fwd_data, rd1_register);
		// 	end
		// end

		assign out_rd2 = (in_src2_fwd == 1'b1) ?  in_src2_fwd_data : rd2_register;



		assign out_is_csr   = is_csr;
    	assign out_csr_mask = (is_csr_immed == 1'b1) ?  {27'h0, out_rs1} : out_rd1;


		assign out_wb       = (is_jal || is_jalr || is_e_inst) ? `WB_JAL :
			                   is_linst ? `WB_MEM :
			                   	     (is_itype || is_rtype || is_lui || is_auipc || is_csr) ?  `WB_ALU :
			                   	     	    `NO_WB;


		assign out_rs2_src   = (is_itype || is_stype) ? `RS2_IMMED : `RS2_REG;

		// MEM signals 
		assign out_mem_read  = (is_linst) ? func3 : `NO_MEM_READ;
		assign out_mem_write = (is_stype) ? func3 : `NO_MEM_WRITE;

		// UPPER IMMEDIATE
		always @(*) begin
			case(curr_opcode)
				`LUI_INST:   out_upper_immed  = {func7, out_rs2, out_rs1, func3};
				`AUIPC_INST: out_upper_immed  = {func7, out_rs2, out_rs1, func3};
				default:    out_upper_immed  = 20'h0;
			endcase // curr_opcode
		end


		assign jal_b_19_to_12      = in_instruction[19:12];
		assign jal_b_11            = in_instruction[20];
		assign jal_b_10_to_1       = in_instruction[30:21];
		assign jal_b_20            = in_instruction[31];
		assign jal_b_0             = 1'b0;
		assign jal_unsigned_offset = {jal_b_20, jal_b_19_to_12, jal_b_11, jal_b_10_to_1, jal_b_0};
		assign jal_1_offset         = {{11{jal_b_20}}, jal_unsigned_offset};


		assign jalr_immed   = {func7, out_rs2};
		assign jal_2_offset = {{20{jalr_immed[11]}}, jalr_immed};


		assign jal_sys_cond1 = func3 == 3'h0;
		assign jal_sys_cond2 = u_12  < 12'h2;

		assign jal_sys_jal = (jal_sys_cond1 && jal_sys_cond2) ? 1'b1 : 1'b0;
		assign jal_sys_off = (jal_sys_cond1 && jal_sys_cond2) ? 32'hb0000000 : 32'hdeadbeef;

		// JAL 
		always @(*) begin
			case(curr_opcode)
				`JAL_INST:
					begin
		       		 	out_jal        = 1'b1;
						out_jal_offset = jal_1_offset;
					end
				`JALR_INST:
					begin
		        		out_jal        = 1'b1;
						out_jal_offset = jal_2_offset;
					end
				`SYS_INST:
					begin
						out_jal        = jal_sys_jal;
						out_jal_offset = jal_sys_off;
					end
				default:
					begin
						out_jal          = 1'b0;
						out_jal_offset   = 32'hdeadbeef;
					end
			endcase
		end


		// CSR

		assign csr_cond1  = func3 != 3'h0;
		assign csr_cond2  = u_12  >= 12'h2;

		assign out_csr_address = (csr_cond1 && csr_cond2) ? u_12 : 12'h55;


		// ITYPE IMEED
		assign alu_shift_i       = (func3 == 3'h1) || (func3 == 3'h5);
		assign alu_shift_i_immed = {{7{1'b0}}, out_rs2};
		assign alu_tempp = alu_shift_i ? alu_shift_i_immed : u_12;


		always @(*) begin
			case(curr_opcode)
					`ALU_INST: out_itype_immed = {{20{alu_tempp[11]}}, alu_tempp};
					`S_INST:   out_itype_immed = {{20{func7[6]}}, func7, out_rd};
					`L_INST:   out_itype_immed = {{20{u_12[11]}}, u_12};
					`B_INST:   out_itype_immed = {{20{in_instruction[31]}}, in_instruction[31], in_instruction[7], in_instruction[30:25], in_instruction[11:8]};
					default:  out_itype_immed = 32'hdeadbeef;
				endcase
		end
	

		always @(*) begin
			case(curr_opcode)
				`B_INST:
					begin
						out_branch_stall = 1'b1;
						case(func3)
							3'h0: out_branch_type = `BEQ;
							3'h1: out_branch_type = `BNE;
							3'h4: out_branch_type = `BLT;
							3'h5: out_branch_type = `BGT;
							3'h6: out_branch_type = `BLTU;
							3'h7: out_branch_type = `BGTU;
							default: out_branch_type = `NO_BRANCH; 
						endcase
					end

				`JAL_INST:
					begin
						out_branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b1;
					end
				`JALR_INST:
					begin
						out_branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b1;
					end
				default:
					begin
						out_branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b0;
					end
			endcase
		end


		always @(*) begin
			// ALU OP
			case(func3)
				3'h0:    alu_op = (curr_opcode == `ALU_INST) ? `ADD : (func7 == 7'h0 ? `ADD : `SUB);
				3'h1:    alu_op = `SLLA;
				3'h2:    alu_op = `SLT;
				3'h3:    alu_op = `SLTU;
				3'h4:    alu_op = `XOR;
				3'h5:    alu_op = (func7 == 7'h0) ? `SRL : `SRA;
				3'h6:    alu_op = `OR;
				3'h7:    alu_op = `AND;
				default: alu_op = `NO_ALU; 
			endcase
		end

		always @(*) begin
			// ALU OP
			case(func3)
				3'h0:    mul_alu = `MUL;
				3'h1:    mul_alu = `MULH;
				3'h2:    mul_alu = `MULHSU;
				3'h3:    mul_alu = `MULHU;
				3'h4:    mul_alu = `DIV;
				3'h5:    mul_alu = `DIVU;
				3'h6:    mul_alu = `REM;
				3'h7:    mul_alu = `REMU;
				default: mul_alu = `NO_ALU; 
			endcase
		end

		assign csr_type = func3[1:0];

		always @(*) begin
			case(csr_type)
				2'h1:     csr_alu = `CSR_ALU_RW;
				2'h2:     csr_alu = `CSR_ALU_RS;
				2'h3:     csr_alu = `CSR_ALU_RC;
				default:  csr_alu = `NO_ALU;
			endcase
		end

		wire[4:0] temp_final_alu;

		assign temp_final_alu = is_btype ? ((out_branch_type < `BLTU) ? `SUB : `SUBU) :
										is_lui ? `LUI_ALU :
											is_auipc ? `AUIPC_ALU :
												is_csr ? csr_alu :
													(is_stype || is_linst) ? `ADD :
														alu_op;

		assign out_alu_op = ((func7[0] == 1'b1) && is_rtype) ? mul_alu : temp_final_alu;

endmodule








