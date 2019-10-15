
`include "VX_define.v"

module VX_decode(
	// Fetch Inputs
	VX_inst_meta_inter     fd_inst_meta_de,

	// WriteBack inputs
	// VX_wb_inter            VX_writeback_inter,


	// Fwd Request
	// VX_forward_reqeust_inter VX_fwd_req_de,

	// FORWARDING INPUTS
	// VX_forward_response_inter   VX_fwd_rsp,

	// input wire[`NW_M1:0]          in_which_wspawn,

	// Outputs
	VX_frE_to_bckE_req_inter VX_frE_to_bckE_req,
	output reg               out_gpr_stall,
	output reg               out_branch_stall,
	output wire              out_ebreak

);

		assign out_gpr_stall = 0;

		wire[31:0]      in_instruction     = fd_inst_meta_de.instruction;
		wire[31:0]      in_curr_PC         = fd_inst_meta_de.inst_pc;
		wire[`NW_M1:0]  in_warp_num        = fd_inst_meta_de.warp_num;

		assign VX_frE_to_bckE_req.curr_PC  = in_curr_PC;

		wire[`NT_M1:0]  in_valid = fd_inst_meta_de.valid;

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
		// wire       is_clone;
		wire       is_jalrs;
		wire       is_jmprt;
		wire       is_wspawn;

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



		// assign VX_fwd_req_de.src1        = VX_frE_to_bckE_req.rs1;
		// assign VX_fwd_req_de.src2        = VX_frE_to_bckE_req.rs2;
		// assign VX_fwd_req_de.warp_num    = VX_frE_to_bckE_req.warp_num;


		// VX_gpr_read_inter VX_gpr_read();
		// assign VX_gpr_read.rs1      = VX_frE_to_bckE_req.rs1;
		// assign VX_gpr_read.rs2      = VX_frE_to_bckE_req.rs2;
		// assign VX_gpr_read.warp_num = VX_frE_to_bckE_req.warp_num;

		// VX_gpr_jal_inter VX_gpr_jal();
		// assign VX_gpr_jal.is_jal  = is_jal;
		// assign VX_gpr_jal.curr_PC = in_curr_PC;


		// VX_gpr_clone_inter VX_gpr_clone();
		// assign VX_gpr_clone.is_clone = is_clone;
		// assign VX_gpr_clone.warp_num = VX_frE_to_bckE_req.warp_num;


		// VX_gpr_wspawn_inter VX_gpr_wspawn();
		// assign VX_gpr_wspawn.is_wspawn    = is_wspawn;
		// assign VX_gpr_wspawn.which_wspawn = in_which_wspawn;
		// // assign VX_gpr_wspawn.warp_num     = VX_frE_to_bckE_req.warp_num;

		// VX_gpr_wrapper vx_grp_wrapper(
		// 		.clk            (clk),
		// 		.VX_writeback_inter(VX_writeback_inter),
		// 		.VX_fwd_rsp        (VX_fwd_rsp),
		// 		.VX_gpr_read       (VX_gpr_read),
		// 		.VX_gpr_jal        (VX_gpr_jal),
		// 		.VX_gpr_clone      (VX_gpr_clone),
		// 		.VX_gpr_wspawn     (VX_gpr_wspawn),

		// 		.out_a_reg_data (VX_frE_to_bckE_req.a_reg_data),
		// 		.out_b_reg_data (VX_frE_to_bckE_req.b_reg_data),
		// 		.out_gpr_stall(out_gpr_stall)
		// 	);





		assign VX_frE_to_bckE_req.valid    = fd_inst_meta_de.valid;

		assign VX_frE_to_bckE_req.warp_num = in_warp_num;


		assign curr_opcode    = in_instruction[6:0];


		assign VX_frE_to_bckE_req.rd   = in_instruction[11:7];
		assign VX_frE_to_bckE_req.rs1  = in_instruction[19:15];
		assign VX_frE_to_bckE_req.rs2  = in_instruction[24:20];
		assign func3    = in_instruction[14:12];
		assign func7    = in_instruction[31:25];
		assign u_12     = in_instruction[31:20];


		assign VX_frE_to_bckE_req.PC_next = in_curr_PC + 32'h4;


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

		assign is_gpgpu     = (curr_opcode == `GPGPU_INST);
		// assign is_clone     = is_gpgpu && (func3 == 5);
		assign is_jalrs     = is_gpgpu && (func3 == 6);
		assign is_jmprt     = is_gpgpu && (func3 == 4);
		assign is_wspawn    = is_gpgpu && (func3 == 0);


		assign VX_frE_to_bckE_req.csr_immed = is_csr_immed;
		assign VX_frE_to_bckE_req.wspawn    = is_wspawn;



		// wire[`NT_M1:0] jalrs_thread_mask = 0;
		// wire[`NT_M1:0] jmprt_thread_mask;

		// genvar tm_i;
		// generate
		// 	for (tm_i = 0; tm_i < `NT; tm_i = tm_i + 1) begin
		// 			assign jalrs_thread_mask[tm_i] = $signed(tm_i) <= $signed(VX_frE_to_bckE_req.b_reg_data[0]);
		// 	end
		// endgenerate


		// genvar tm_ji;
		// generate
		// 	assign jmprt_thread_mask[0] = 1;
		// 	for (tm_ji = 1; tm_ji < `NT; tm_ji = tm_ji + 1) begin
		// 			assign jmprt_thread_mask[tm_ji] = 0;
		// 	end
		// endgenerate


		assign VX_frE_to_bckE_req.wb       = (is_jal || is_jalr || is_jalrs || is_e_inst) ? `WB_JAL :
			                   					is_linst ? `WB_MEM :
			                   	     				(is_itype || is_rtype || is_lui || is_auipc || is_csr) ?  `WB_ALU :
			                   	     	    			`NO_WB;


		assign VX_frE_to_bckE_req.rs2_src   = (is_itype || is_stype) ? `RS2_IMMED : `RS2_REG;

		// MEM signals 
		assign VX_frE_to_bckE_req.mem_read  = (is_linst) ? func3 : `NO_MEM_READ;
		assign VX_frE_to_bckE_req.mem_write = (is_stype) ? func3 : `NO_MEM_WRITE;

		// UPPER IMMEDIATE
		reg[19:0] temp_upper_immed;
		always @(*) begin
			case(curr_opcode)
				`LUI_INST:   temp_upper_immed  = {func7, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.rs1, func3};
				`AUIPC_INST: temp_upper_immed  = {func7, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.rs1, func3};
				default:     temp_upper_immed  = 20'h0;
			endcase // curr_opcode
		end

		assign VX_frE_to_bckE_req.upper_immed = temp_upper_immed;


		assign jal_b_19_to_12      = in_instruction[19:12];
		assign jal_b_11            = in_instruction[20];
		assign jal_b_10_to_1       = in_instruction[30:21];
		assign jal_b_20            = in_instruction[31];
		assign jal_b_0             = 1'b0;
		assign jal_unsigned_offset = {jal_b_20, jal_b_19_to_12, jal_b_11, jal_b_10_to_1, jal_b_0};
		assign jal_1_offset         = {{11{jal_b_20}}, jal_unsigned_offset};


		assign jalr_immed   = {func7, VX_frE_to_bckE_req.rs2};
		assign jal_2_offset = {{20{jalr_immed[11]}}, jalr_immed};


		assign jal_sys_cond1 = func3 == 3'h0;
		assign jal_sys_cond2 = u_12  < 12'h2;

		assign jal_sys_jal = (jal_sys_cond1 && jal_sys_cond2) ? 1'b1 : 1'b0;
		assign jal_sys_off = (jal_sys_cond1 && jal_sys_cond2) ? 32'hb0000000 : 32'hdeadbeef;

		// JAL 
		reg       temp_jal;
		reg[31:0] temp_jal_offset;
		always @(*) begin
			case(curr_opcode)
				`JAL_INST:
					begin
		       		 	temp_jal        = 1'b1 && in_valid[0];
						temp_jal_offset = jal_1_offset;
					end
				`JALR_INST:
					begin
		        		temp_jal        = 1'b1 && in_valid[0];
						temp_jal_offset = jal_2_offset;
					end
				`GPGPU_INST:
					begin
						if (is_jalrs || is_jmprt)
						begin
			        		temp_jal        = 1'b1 && in_valid[0];
							temp_jal_offset = 32'h0;
						end
					end
				`SYS_INST:
					begin
						// $display("SYS EBREAK %h", (jal_sys_jal && in_valid[0]) );
						temp_jal        = jal_sys_jal && in_valid[0];
						temp_jal_offset = jal_sys_off;
					end
				default:
					begin
						temp_jal          = 1'b0 && in_valid[0];
						temp_jal_offset   = 32'hdeadbeef;
					end
			endcase
		end

		assign VX_frE_to_bckE_req.jalQual    = is_jal;
		assign VX_frE_to_bckE_req.jal        = temp_jal;
		assign VX_frE_to_bckE_req.jal_offset = temp_jal_offset;

		// wire is_ebreak;


		// assign is_ebreak = is_e_inst;
		wire ebreak = (curr_opcode == `SYS_INST) && (jal_sys_jal && in_valid[0]);
		assign VX_frE_to_bckE_req.ebreak = ebreak;
		assign out_ebreak = ebreak;



		// CSR

		assign csr_cond1  = func3 != 3'h0;
		assign csr_cond2  = u_12  >= 12'h2;

		assign VX_frE_to_bckE_req.csr_address = (csr_cond1 && csr_cond2) ? u_12 : 12'h55;


		// ITYPE IMEED
		assign alu_shift_i       = (func3 == 3'h1) || (func3 == 3'h5);
		assign alu_shift_i_immed = {{7{1'b0}}, VX_frE_to_bckE_req.rs2};
		assign alu_tempp = alu_shift_i ? alu_shift_i_immed : u_12;


		reg[31:0] temp_itype_immed;
		always @(*) begin
			case(curr_opcode)
					`ALU_INST: temp_itype_immed = {{20{alu_tempp[11]}}, alu_tempp};
					`S_INST:   temp_itype_immed = {{20{func7[6]}}, func7, VX_frE_to_bckE_req.rd};
					`L_INST:   temp_itype_immed = {{20{u_12[11]}}, u_12};
					`B_INST:   temp_itype_immed = {{20{in_instruction[31]}}, in_instruction[31], in_instruction[7], in_instruction[30:25], in_instruction[11:8]};
					default:   temp_itype_immed = 32'hdeadbeef;
				endcase
		end

		assign VX_frE_to_bckE_req.itype_immed = temp_itype_immed;
	

		reg[2:0] temp_branch_type;
		reg      temp_branch_stall;
		always @(*) begin
			case(curr_opcode)
				`B_INST:
					begin
						// $display("BRANCH IN DECODE");
						temp_branch_stall = 1'b1 && in_valid[0];
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
						temp_branch_stall = 1'b1 && in_valid[0];
					end
				`JALR_INST:
					begin
						temp_branch_type  = `NO_BRANCH;
						temp_branch_stall = 1'b1 && in_valid[0];
					end
				`GPGPU_INST:
					begin
						if (is_jalrs || is_jmprt)
						begin
							temp_branch_type  = `NO_BRANCH;
							temp_branch_stall = 1'b1 && in_valid[0];
						end
					end
				default:
					begin
						temp_branch_type  = `NO_BRANCH;
						temp_branch_stall = 1'b0 && in_valid[0];
					end
			endcase
		end

		assign VX_frE_to_bckE_req.branch_type = temp_branch_type;
		assign out_branch_stall               = temp_branch_stall && in_valid[0];

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

		assign temp_final_alu = is_btype ? ((VX_frE_to_bckE_req.branch_type < `BLTU) ? `SUB : `SUBU) :
										is_lui ? `LUI_ALU :
											is_auipc ? `AUIPC_ALU :
												is_csr ? csr_alu :
													(is_stype || is_linst) ? `ADD :
														alu_op;

		assign VX_frE_to_bckE_req.alu_op = ((func7[0] == 1'b1) && is_rtype) ? mul_alu : temp_final_alu;

endmodule








