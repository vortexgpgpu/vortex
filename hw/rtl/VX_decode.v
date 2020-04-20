
`include "VX_define.v"

module VX_decode(
	// Fetch Inputs
	VX_inst_meta_if       fd_inst_meta_de,

	// Outputs
	VX_frE_to_bckE_req_if frE_to_bckE_req_if,
	VX_wstall_if          wstall_if,
	VX_join_if            join_if,

	output wire           terminate_sim
);

	wire[31:0]      in_instruction      = fd_inst_meta_de.instruction;
	wire[31:0]      in_curr_PC          = fd_inst_meta_de.inst_pc;
	wire[`NW_BITS-1:0]  in_warp_num     = fd_inst_meta_de.warp_num;

	assign frE_to_bckE_req_if.curr_PC   = in_curr_PC;

	wire[`NUM_THREADS-1:0]  in_valid 	= fd_inst_meta_de.valid;

	wire[6:0]  curr_opcode;

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

	wire       is_gpgpu;
	wire       is_wspawn;
	wire       is_tmc;
	wire       is_split;
	wire       is_join;
	wire       is_barrier;

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
	reg[19:0]  temp_upper_immed;
	reg       temp_jal;
	reg[31:0] temp_jal_offset;
	reg[31:0] temp_itype_immed;
	reg[2:0] temp_branch_type;
	reg      temp_branch_stall;

	assign frE_to_bckE_req_if.valid = fd_inst_meta_de.valid;

	assign frE_to_bckE_req_if.warp_num = in_warp_num;

	assign curr_opcode = in_instruction[6:0];

	assign frE_to_bckE_req_if.rd   = in_instruction[11:7];
	assign frE_to_bckE_req_if.rs1  = in_instruction[19:15];
	assign frE_to_bckE_req_if.rs2  = in_instruction[24:20];
	assign func3    = in_instruction[14:12];
	assign func7    = in_instruction[31:25];
	assign u_12     = in_instruction[31:20];

	assign frE_to_bckE_req_if.PC_next = in_curr_PC + 32'h4;

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
	// assign is_e_inst    = (curr_opcode == `SYS_INST) && (func3 == 0);
	assign is_e_inst    = in_instruction == 32'h00000073;

	assign is_gpgpu     = (curr_opcode == `GPGPU_INST);

	assign is_tmc       = is_gpgpu && (func3 == 0); // Goes to BE
	assign is_wspawn    = is_gpgpu && (func3 == 1); // Goes to BE
	assign is_barrier   = is_gpgpu && (func3 == 4); // Goes to BE
	assign is_split     = is_gpgpu && (func3 == 2); // Goes to BE
	assign is_join      = is_gpgpu && (func3 == 3); // Doesn't go to BE

	assign join_if.is_join       = is_join;
	assign join_if.join_warp_num = in_warp_num;

	assign frE_to_bckE_req_if.is_wspawn  = is_wspawn;
	assign frE_to_bckE_req_if.is_tmc     = is_tmc;
	assign frE_to_bckE_req_if.is_split   = is_split;
	assign frE_to_bckE_req_if.is_barrier = is_barrier;

	assign frE_to_bckE_req_if.csr_immed = is_csr_immed;
	assign frE_to_bckE_req_if.is_csr    = is_csr;

	assign frE_to_bckE_req_if.wb       = (is_jal || is_jalr || is_e_inst) ? `WB_JAL :
											is_linst ? `WB_MEM :
												(is_itype || is_rtype || is_lui || is_auipc || is_csr) ?  `WB_ALU :
													`NO_WB;

	assign frE_to_bckE_req_if.rs2_src   = (is_itype || is_stype) ? `RS2_IMMED : `RS2_REG;

	// MEM signals 
	assign frE_to_bckE_req_if.mem_read  = (is_linst) ? func3 : `NO_MEM_READ;
	assign frE_to_bckE_req_if.mem_write = (is_stype) ? func3 : `NO_MEM_WRITE;

	// UPPER IMMEDIATE
	always @(*) begin
		case(curr_opcode)
			`LUI_INST:   temp_upper_immed  = {func7, frE_to_bckE_req_if.rs2, frE_to_bckE_req_if.rs1, func3};
			`AUIPC_INST: temp_upper_immed  = {func7, frE_to_bckE_req_if.rs2, frE_to_bckE_req_if.rs1, func3};
			default:     temp_upper_immed  = 20'h0;
		endcase // curr_opcode
	end

	assign frE_to_bckE_req_if.upper_immed = temp_upper_immed;

	assign jal_b_19_to_12      = in_instruction[19:12];
	assign jal_b_11            = in_instruction[20];
	assign jal_b_10_to_1       = in_instruction[30:21];
	assign jal_b_20            = in_instruction[31];
	assign jal_b_0             = 1'b0;
	assign jal_unsigned_offset = {jal_b_20, jal_b_19_to_12, jal_b_11, jal_b_10_to_1, jal_b_0};
	assign jal_1_offset        = {{11{jal_b_20}}, jal_unsigned_offset};

	assign jalr_immed   = {func7, frE_to_bckE_req_if.rs2};
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
					temp_jal        = 1'b1 && (|in_valid);
					temp_jal_offset = jal_1_offset;
				end
			`JALR_INST:
				begin
					temp_jal        = 1'b1 && (|in_valid);
					temp_jal_offset = jal_2_offset;
				end
			`SYS_INST:
				begin
					// $display("SYS EBREAK %h", (jal_sys_jal && (|in_valid)) );
					temp_jal        = jal_sys_jal && (|in_valid);
					temp_jal_offset = jal_sys_off;
				end
			default:
				begin
					temp_jal          = 1'b0 && (|in_valid);
					temp_jal_offset   = 32'hdeadbeef;
				end
		endcase
	end

	assign frE_to_bckE_req_if.jalQual    = is_jal;
	assign frE_to_bckE_req_if.jal        = temp_jal;
	assign frE_to_bckE_req_if.jal_offset = temp_jal_offset;

	// wire is_ebreak;

	// assign is_ebreak = is_e_inst;
	wire ebreak = (curr_opcode == `SYS_INST) && (jal_sys_jal && (|in_valid));
	assign frE_to_bckE_req_if.ebreak = ebreak;
	assign terminate_sim = is_e_inst;

	// CSR

	assign csr_cond1  = func3 != 3'h0;
	assign csr_cond2  = u_12  >= 12'h2;

	assign frE_to_bckE_req_if.csr_address = (csr_cond1 && csr_cond2) ? u_12 : 12'h55;

	// ITYPE IMEED
	assign alu_shift_i       = (func3 == 3'h1) || (func3 == 3'h5);
	assign alu_shift_i_immed = {{7{1'b0}}, frE_to_bckE_req_if.rs2};
	assign alu_tempp = alu_shift_i ? alu_shift_i_immed : u_12;

	always @(*) begin
		case(curr_opcode)
				`ALU_INST: temp_itype_immed = {{20{alu_tempp[11]}}, alu_tempp};
				`S_INST:   temp_itype_immed = {{20{func7[6]}}, func7, frE_to_bckE_req_if.rd};
				`L_INST:   temp_itype_immed = {{20{u_12[11]}}, u_12};
				`B_INST:   temp_itype_immed = {{20{in_instruction[31]}}, in_instruction[31], in_instruction[7], in_instruction[30:25], in_instruction[11:8]};
				default:   temp_itype_immed = 32'hdeadbeef;
			endcase
	end

	assign frE_to_bckE_req_if.itype_immed = temp_itype_immed;

	always @(*) begin
		case(curr_opcode)
			`B_INST:
				begin
					// $display("BRANCH IN DECODE");
					temp_branch_stall = 1'b1 && (|in_valid);
					case(func3)
						3'h0: temp_branch_type = `BEQ;
						3'h1: temp_branch_type = `BNE;
						3'h4: temp_branch_type = `BLT;
						3'h5: temp_branch_type = `BGT;
						3'h6: temp_branch_type = `BLTU;
						3'h7: temp_branch_type = `BGTU;
						default: temp_branch_type = `NO_BRANCH; 
					endcase
				end

			`JAL_INST:
				begin
					temp_branch_type  = `NO_BRANCH;
					temp_branch_stall = 1'b1 && (|in_valid);
				end
			`JALR_INST:
				begin
					temp_branch_type  = `NO_BRANCH;
					temp_branch_stall = 1'b1 && (|in_valid);
				end
			default:
				begin
					temp_branch_type  = `NO_BRANCH;
					temp_branch_stall = 1'b0 && (|in_valid);
				end
		endcase
	end

	assign frE_to_bckE_req_if.branch_type = temp_branch_type;

	assign wstall_if.wstall               = (temp_branch_stall || is_tmc || is_split || is_barrier) && (|in_valid);
	assign wstall_if.warp_num             = in_warp_num;

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

	assign temp_final_alu = is_btype ? ((frE_to_bckE_req_if.branch_type < `BLTU) ? `SUB : `SUBU) :
								is_lui ? `LUI_ALU :
									is_auipc ? `AUIPC_ALU :
										is_csr ? csr_alu :
											(is_stype || is_linst) ? `ADD :
						              			alu_op;

	assign frE_to_bckE_req_if.alu_op = ((func7[0] == 1'b1) && is_rtype) ? mul_alu : temp_final_alu;

endmodule








