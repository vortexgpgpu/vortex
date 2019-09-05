
`include "VX_define.v"

module VX_decode(
	input wire             clk,
	// Fetch Inputs
	VX_inst_meta_inter     fd_inst_meta_de,

	// WriteBack inputs
	VX_wb_inter            VX_writeback_inter,

	// FORWARDING INPUTS
	input wire            in_src1_fwd,
	input wire[`NT_M1:0][31:0]      in_src1_fwd_data,
	input wire            in_src2_fwd,
	input wire[`NT_M1:0][31:0]      in_src2_fwd_data,
	input wire[`NW_M1:0]  in_which_wspawn,

	// Outputs
	VX_frE_to_bckE_req_inter VX_frE_to_bckE_req,
	VX_warp_ctl_inter        VX_warp_ctl,
	output reg               out_clone_stall,
	output reg               out_branch_stall

);


		wire[`NT_M1:0][31:0]      in_write_data;
		wire[4:0]       in_rd;
		wire[1:0]       in_wb;
		wire[`NT_M1:0]  in_wb_valid;
		wire[`NW_M1:0]  in_wb_warp_num;


		assign in_write_data  = VX_writeback_inter.write_data;
		assign in_rd          = VX_writeback_inter.rd;
		assign in_wb          = VX_writeback_inter.wb;
		assign in_wb_valid    = VX_writeback_inter.wb_valid;
		assign in_wb_warp_num = VX_writeback_inter.wb_warp_num;

		wire[31:0]      in_instruction     = fd_inst_meta_de.instruction;
		wire[31:0]      in_curr_PC         = fd_inst_meta_de.inst_pc;
		wire[`NW_M1:0]  in_warp_num        = fd_inst_meta_de.warp_num;

		assign VX_frE_to_bckE_req.curr_PC  = in_curr_PC;

		wire            in_valid[`NT_M1:0];
		genvar index;
		for (index = 0; index <= `NT_M1; index = index + 1) assign in_valid[index] = fd_inst_meta_de.valid[index];

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
		wire       is_clone;
		wire       is_jalrs;
		wire       is_jmprt;
		wire       is_wspawn;

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

		/* verilator lint_off UNUSED */
		wire[31:0][31:0] w0_t0_registers;
		/* verilator lint_on UNUSED */



		// always @(*) begin
		// 	$display("DECODE WARP: %h", in_warp_num);
		// end


		// always @(*) begin
		// 	$display("DECODE WARP: %h PC: %h",in_warp_num, in_curr_PC);
		// end

		`ifdef ONLY

			wire[31:0] glob_a_reg_data[`NT_M1:0];
			wire[31:0] glob_b_reg_data[`NT_M1:0];
			reg        glob_clone_stall;

			wire       curr_warp_zero     = in_warp_num == 0;
			wire       context_zero_valid = (in_wb_warp_num == 0);
			wire       real_zero_isclone  = is_clone  && (in_warp_num == 0);  
			VX_context VX_Context_zero(
				.clk              (clk),
				.in_warp          (curr_warp_zero),
				.in_wb_warp       (context_zero_valid),
				.in_valid         (in_wb_valid),
				.in_rd            (in_rd),
				.in_src1          (VX_frE_to_bckE_req.rs1),
				.in_src2          (VX_frE_to_bckE_req.rs2),
				.in_curr_PC       (in_curr_PC),
				.in_is_clone      (real_zero_isclone),
				.in_is_jal        (is_jal),
				.in_src1_fwd      (in_src1_fwd),
				.in_src1_fwd_data (in_src1_fwd_data),
				.in_src2_fwd      (in_src2_fwd),
				.in_src2_fwd_data (in_src2_fwd_data),
				.in_write_register(write_register),
				.in_write_data    (in_write_data),
				.out_a_reg_data   (glob_a_reg_data),
				.out_b_reg_data   (glob_b_reg_data),
				.out_clone_stall  (glob_clone_stall),
				.w0_t0_registers  (w0_t0_registers)
			);


			assign VX_frE_to_bckE_req.a_reg_data  = glob_a_reg_data;
			assign VX_frE_to_bckE_req.b_reg_data  = glob_b_reg_data;
			assign out_clone_stall = glob_clone_stall;

		`else 

			wire[`NW-1:0][`NT_M1:0][31:0] glob_a_reg_data;
			wire[`NW-1:0][`NT_M1:0][31:0] glob_b_reg_data;
			reg[`NW-1:0]        glob_clone_stall;

			wire       curr_warp_zero     = in_warp_num == 0;
			wire       context_zero_valid = (in_wb_warp_num == 0);
			wire       real_zero_isclone  = is_clone  && (in_warp_num == 0);  
			VX_context VX_Context_zero(
				.clk              (clk),
				.in_warp          (curr_warp_zero),
				.in_wb_warp       (context_zero_valid),
				.in_valid         (in_wb_valid),
				.in_rd            (in_rd),
				.in_src1          (VX_frE_to_bckE_req.rs1),
				.in_src2          (VX_frE_to_bckE_req.rs2),
				.in_curr_PC       (in_curr_PC),
				.in_is_clone      (real_zero_isclone),
				.in_is_jal        (is_jal),
				.in_src1_fwd      (in_src1_fwd),
				.in_src1_fwd_data (in_src1_fwd_data),
				.in_src2_fwd      (in_src2_fwd),
				.in_src2_fwd_data (in_src2_fwd_data),
				.in_write_register(write_register),
				.in_write_data    (in_write_data),
				.out_a_reg_data   (glob_a_reg_data[0]),
				.out_b_reg_data   (glob_b_reg_data[0]),
				.out_clone_stall  (glob_clone_stall[0]),
				.w0_t0_registers  (w0_t0_registers)
			);

			genvar r;
			generate
				for (r = 1; r < `NW; r = r + 1) begin
					wire context_glob_valid = (in_wb_warp_num == r);
					wire curr_warp_glob     = in_warp_num == r;
					wire real_wspawn        = is_wspawn && (in_which_wspawn == r); 
					wire real_isclone       = is_clone  && (in_warp_num == r);      
					VX_context_slave VX_Context_one(
						.clk              (clk),
						.in_warp          (curr_warp_glob),
						.in_wb_warp       (context_glob_valid),
						.in_valid         (in_wb_valid),
						.in_rd            (in_rd),
						.in_src1          (VX_frE_to_bckE_req.rs1),
						.in_src2          (VX_frE_to_bckE_req.rs2),
						.in_curr_PC       (in_curr_PC),
						.in_is_clone      (real_isclone),
						.in_is_jal        (is_jal),
						.in_src1_fwd      (in_src1_fwd),
						.in_src1_fwd_data (in_src1_fwd_data),
						.in_src2_fwd      (in_src2_fwd),
						.in_src2_fwd_data (in_src2_fwd_data),
						.in_write_register(write_register),
						.in_write_data    (in_write_data),
						.in_wspawn_regs   (w0_t0_registers),
						.in_wspawn        (real_wspawn),
						.out_a_reg_data   (glob_a_reg_data[r]),
						.out_b_reg_data   (glob_b_reg_data[r]),
						.out_clone_stall  (glob_clone_stall[r])
					);
				end
			endgenerate

			// always @(posedge clk)
			// 	if(write_register && (in_wb_warp == 3) && (in_wb_valid[0]) && (in_rd == 31)) begin

			// 		$display("Warp 3 writing ",);
			// 	end
			// end

			reg[`NT_M1:0][31:0] temp_out_a_reg_data;
			reg[`NT_M1:0][31:0] temp_out_b_reg_data;
			/* verilator lint_off UNOPTFLAT */
			reg       temp_out_clone_stall;
			/* verilator lint_on UNOPTFLAT */

			always @(*) begin

				if (`NW == 1) begin
					temp_out_a_reg_data = glob_a_reg_data[0];
					temp_out_b_reg_data = glob_b_reg_data[0];
				end else begin
					integer g;
					// temp_out_clone_stall = 0;
					for (g = 0; g < `NW; g = g + 1)
					begin
						if (in_warp_num == g[`NW_M1:0]) begin
							temp_out_a_reg_data = glob_a_reg_data[g];
							temp_out_b_reg_data = glob_b_reg_data[g];
						end

						// temp_out_clone_stall = temp_out_clone_stall || glob_clone_stall[g];
					end
				end
			end

			genvar sml_index;
			for (sml_index = 0; sml_index <= `NT_M1; sml_index = sml_index + 1) begin
				assign VX_frE_to_bckE_req.a_reg_data[sml_index]  = temp_out_a_reg_data[sml_index];
				assign VX_frE_to_bckE_req.b_reg_data[sml_index]  = temp_out_b_reg_data[sml_index];
			end

			// assign out_clone_stall = temp_out_clone_stall;

			// assign out_a_reg_data  = curr_warp_zero ? glob_a_reg_data[0]  : glob_a_reg_data[1];
			// assign out_b_reg_data  = curr_warp_zero ? glob_b_reg_data[0]  : glob_b_reg_data[1];

			genvar y;
			generate
				always @(*) begin
					temp_out_clone_stall = glob_clone_stall[0];
					for (y = 1; y < `NW; y = y+1) begin
						temp_out_clone_stall = temp_out_clone_stall || glob_clone_stall[y];
					end
				end
			endgenerate

			assign out_clone_stall = temp_out_clone_stall;


		`endif


		// assign out_clone_stall = glob_clone_stall[0] || glob_clone_stall[1] || 
		//                          glob_clone_stall[2] || glob_clone_stall[3];

		// always @(*) begin
		// 	if (context_one_valid) begin
		// 		$display("PC: %h -> src1: %h\tsrc2: %h",in_curr_PC, one_a_reg_data[0], one_b_reg_data[0]);
		// 	end
		// end


		assign VX_frE_to_bckE_req.valid    = fd_inst_meta_de.valid;

		assign VX_frE_to_bckE_req.warp_num = in_warp_num;
		assign VX_warp_ctl.warp_num        = in_warp_num;

		assign write_register = (in_wb != 2'h0) ? (1'b1) : (1'b0);


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
		assign is_clone     = is_gpgpu && (func3 == 5);
		assign is_jalrs     = is_gpgpu && (func3 == 6);
		assign is_jmprt     = is_gpgpu && (func3 == 4);
		assign is_wspawn    = is_gpgpu && (func3 == 0);

		assign VX_warp_ctl.wspawn    = is_wspawn;
		assign VX_warp_ctl.wspawn_pc = VX_frE_to_bckE_req.a_reg_data[0];




		wire[`NT_M1:0] jalrs_thread_mask;
		wire[`NT_M1:0] jmprt_thread_mask;

		genvar tm_i;
		generate
			for (tm_i = 0; tm_i < `NT; tm_i = tm_i + 1) begin
					/* verilator lint_off UNSIGNED */
					assign jalrs_thread_mask[tm_i] = tm_i <= $signed(VX_frE_to_bckE_req.b_reg_data[0]);
					/* verilator lint_on UNSIGNED */
			end
		endgenerate


		genvar tm_ji;
		generate
			assign jmprt_thread_mask[0] = 1;
			for (tm_ji = 1; tm_ji < `NT; tm_ji = tm_ji + 1) begin
					assign jmprt_thread_mask[tm_ji] = 0;
			end
		endgenerate

		assign VX_warp_ctl.thread_mask = is_jalrs ? jalrs_thread_mask : jmprt_thread_mask;


		assign VX_warp_ctl.change_mask = is_jalrs || is_jmprt;




		// assign out_clone    = is_clone;
		// always @(in_instruction) begin
		// 	$display("Decode inst: %h", in_instruction);
		// end




		// assign out_reg_data[0]   = (    (is_jal == 1'b1) ? in_curr_PC : ((in_src1_fwd == 1'b1) ? in_src1_fwd_data[0] : rd1_register[0]));
		// assign out_reg_data[1]   = (in_src2_fwd == 1'b1) ?  in_src2_fwd_data[0] : rd2_register[0];


		// assign out_reg_data[2]   = (    (is_jal == 1'b1) ? in_curr_PC : ((in_src1_fwd == 1'b1) ? in_src1_fwd_data[1] : rd1_register[1]));
		// assign out_reg_data[3]   = (in_src2_fwd == 1'b1) ?  in_src2_fwd_data[1] : rd2_register[1];

		// assign internal_rd1 = ((is_jal == 1'b1) ? in_curr_PC : ((in_src1_fwd == 1'b1) ? in_src1_fwd_data : rd1_register));
		// assign internal_rd2 = (in_src2_fwd == 1'b1) ?  in_src2_fwd_data : rd2_register;


		// assign out_reg_data[0] = internal_rd1;
		// assign out_reg_data[1] = internal_rd2;


		// always @(negedge clk) begin
		// 	if (in_curr_PC == 32'h800001f0) begin
		// 		$display("IN DECODE: Going to write to: %d with val: %h [%h, %h, %h]", VX_frE_to_bckE_req.rd, internal_rd1, in_curr_PC, in_src1_fwd_data, rd1_register);
		// 	end
		// end

		assign VX_frE_to_bckE_req.is_csr   = is_csr;
    	assign VX_frE_to_bckE_req.csr_mask = (is_csr_immed == 1'b1) ?  {27'h0, VX_frE_to_bckE_req.rs1} : VX_frE_to_bckE_req.a_reg_data[0];


		assign VX_frE_to_bckE_req.wb       = (is_jal || is_jalr || is_jalrs || is_e_inst) ? `WB_JAL :
			                   					is_linst ? `WB_MEM :
			                   	     				(is_itype || is_rtype || is_lui || is_auipc || is_csr) ?  `WB_ALU :
			                   	     	    			`NO_WB;


		assign VX_frE_to_bckE_req.rs2_src   = (is_itype || is_stype) ? `RS2_IMMED : `RS2_REG;

		// MEM signals 
		assign VX_frE_to_bckE_req.mem_read  = (is_linst) ? func3 : `NO_MEM_READ;
		assign VX_frE_to_bckE_req.mem_write = (is_stype) ? func3 : `NO_MEM_WRITE;

		// UPPER IMMEDIATE
		always @(*) begin
			case(curr_opcode)
				`LUI_INST:   VX_frE_to_bckE_req.upper_immed  = {func7, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.rs1, func3};
				`AUIPC_INST: VX_frE_to_bckE_req.upper_immed  = {func7, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.rs1, func3};
				default:     VX_frE_to_bckE_req.upper_immed  = 20'h0;
			endcase // curr_opcode
		end


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
		always @(*) begin
			case(curr_opcode)
				`JAL_INST:
					begin
		       		 	VX_frE_to_bckE_req.jal        = 1'b1 && in_valid[0];
						VX_frE_to_bckE_req.jal_offset = jal_1_offset;
					end
				`JALR_INST:
					begin
		        		VX_frE_to_bckE_req.jal        = 1'b1 && in_valid[0];
						VX_frE_to_bckE_req.jal_offset = jal_2_offset;
					end
				`GPGPU_INST:
					begin
						if (is_jalrs || is_jmprt)
						begin
			        		VX_frE_to_bckE_req.jal        = 1'b1 && in_valid[0];
							VX_frE_to_bckE_req.jal_offset = 32'h0;
						end
					end
				`SYS_INST:
					begin
						// $display("SYS EBREAK %h", (jal_sys_jal && in_valid[0]) );
						VX_frE_to_bckE_req.jal        = jal_sys_jal && in_valid[0];
						VX_frE_to_bckE_req.jal_offset = jal_sys_off;
					end
				default:
					begin
						VX_frE_to_bckE_req.jal          = 1'b0 && in_valid[0];
						VX_frE_to_bckE_req.jal_offset   = 32'hdeadbeef;
					end
			endcase
		end

		wire is_ebreak;


		assign is_ebreak = (curr_opcode == `SYS_INST) && (jal_sys_jal && in_valid[0]);


		assign VX_warp_ctl.ebreak = is_ebreak;

		// CSR

		assign csr_cond1  = func3 != 3'h0;
		assign csr_cond2  = u_12  >= 12'h2;

		assign VX_frE_to_bckE_req.csr_address = (csr_cond1 && csr_cond2) ? u_12 : 12'h55;


		// ITYPE IMEED
		assign alu_shift_i       = (func3 == 3'h1) || (func3 == 3'h5);
		assign alu_shift_i_immed = {{7{1'b0}}, VX_frE_to_bckE_req.rs2};
		assign alu_tempp = alu_shift_i ? alu_shift_i_immed : u_12;


		always @(*) begin
			case(curr_opcode)
					`ALU_INST: VX_frE_to_bckE_req.itype_immed = {{20{alu_tempp[11]}}, alu_tempp};
					`S_INST:   VX_frE_to_bckE_req.itype_immed = {{20{func7[6]}}, func7, VX_frE_to_bckE_req.rd};
					`L_INST:   VX_frE_to_bckE_req.itype_immed = {{20{u_12[11]}}, u_12};
					`B_INST:   VX_frE_to_bckE_req.itype_immed = {{20{in_instruction[31]}}, in_instruction[31], in_instruction[7], in_instruction[30:25], in_instruction[11:8]};
					default:   VX_frE_to_bckE_req.itype_immed = 32'hdeadbeef;
				endcase
		end
	

		always @(*) begin
			case(curr_opcode)
				`B_INST:
					begin
						out_branch_stall = 1'b1 && in_valid[0];
						case(func3)
							3'h0: VX_frE_to_bckE_req.branch_type = `BEQ;
							3'h1: VX_frE_to_bckE_req.branch_type = `BNE;
							3'h4: VX_frE_to_bckE_req.branch_type = `BLT;
							3'h5: VX_frE_to_bckE_req.branch_type = `BGT;
							3'h6: VX_frE_to_bckE_req.branch_type = `BLTU;
							3'h7: VX_frE_to_bckE_req.branch_type = `BGTU;
							default: VX_frE_to_bckE_req.branch_type = `NO_BRANCH; 
						endcase
					end

				`JAL_INST:
					begin
						VX_frE_to_bckE_req.branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b1 && in_valid[0];
					end
				`JALR_INST:
					begin
						VX_frE_to_bckE_req.branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b1 && in_valid[0];
					end
				`GPGPU_INST:
					begin
						if (is_jalrs || is_jmprt)
						begin
							VX_frE_to_bckE_req.branch_type  = `NO_BRANCH;
							out_branch_stall = 1'b1 && in_valid[0];
						end
					end
				default:
					begin
						VX_frE_to_bckE_req.branch_type  = `NO_BRANCH;
						out_branch_stall = 1'b0 && in_valid[0];
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

		assign temp_final_alu = is_btype ? ((VX_frE_to_bckE_req.branch_type < `BLTU) ? `SUB : `SUBU) :
										is_lui ? `LUI_ALU :
											is_auipc ? `AUIPC_ALU :
												is_csr ? csr_alu :
													(is_stype || is_linst) ? `ADD :
														alu_op;

		assign VX_frE_to_bckE_req.alu_op = ((func7[0] == 1'b1) && is_rtype) ? mul_alu : temp_final_alu;

endmodule








