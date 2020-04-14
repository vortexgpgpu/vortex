`include "VX_define.v"

module VX_gpr_stage (
	input wire                 clk,
	input wire                 reset,
	input wire                 schedule_delay,

	input  wire                memory_delay,
	input  wire  			   exec_delay,
	input  wire                stall_gpr_csr,
	output wire                gpr_stage_delay,

	// inputs
		// Instruction Information
	VX_frE_to_bckE_req_inter   VX_bckE_req,

		// WriteBack inputs
	VX_wb_inter                VX_writeback_inter,




	// Outputs
	VX_exec_unit_req_inter   VX_exec_unit_req,
	VX_lsu_req_inter         VX_lsu_req,
	VX_gpu_inst_req_inter    VX_gpu_inst_req,
	VX_csr_req_inter         VX_csr_req
);


	wire[31:0] curr_PC = VX_bckE_req.curr_PC;
	wire[2:0] branchType = VX_bckE_req.branch_type;

	wire is_store = (VX_bckE_req.mem_write != `NO_MEM_WRITE);
	wire is_load  = (VX_bckE_req.mem_read  != `NO_MEM_READ);


	wire jalQual = VX_bckE_req.jalQual;

	VX_gpr_read_inter VX_gpr_read();
	assign VX_gpr_read.rs1      = VX_bckE_req.rs1;
	assign VX_gpr_read.rs2      = VX_bckE_req.rs2;
	assign VX_gpr_read.warp_num = VX_bckE_req.warp_num;

	`ifndef ASIC
		VX_gpr_jal_inter VX_gpr_jal();
		assign VX_gpr_jal.is_jal  = VX_bckE_req.jalQual;
		assign VX_gpr_jal.curr_PC = VX_bckE_req.curr_PC;
	`else 
		VX_gpr_jal_inter VX_gpr_jal();
		assign VX_gpr_jal.is_jal  = VX_exec_unit_req.jalQual;
		assign VX_gpr_jal.curr_PC = VX_exec_unit_req.curr_PC;
	`endif


	VX_gpr_data_inter           VX_gpr_datf();


	VX_gpr_wrapper vx_grp_wrapper(
			.clk               (clk),
			.reset             (reset),
			.VX_writeback_inter(VX_writeback_inter),
			.VX_gpr_read       (VX_gpr_read),
			.VX_gpr_jal        (VX_gpr_jal),

			.out_a_reg_data (VX_gpr_datf.a_reg_data),
			.out_b_reg_data (VX_gpr_datf.b_reg_data)
		);

	// assign VX_bckE_req.is_csr   = is_csr;
	// assign VX_bckE_req_out.csr_mask = (VX_bckE_req.sr_immed == 1'b1) ?  {27'h0, VX_bckE_req.rs1} : VX_gpr_data.a_reg_data[0];

	// Outputs
	VX_exec_unit_req_inter   VX_exec_unit_req_temp();
	VX_lsu_req_inter         VX_lsu_req_temp();
	VX_gpu_inst_req_inter    VX_gpu_inst_req_temp();
	VX_csr_req_inter         VX_csr_req_temp();

	VX_inst_multiplex VX_inst_mult(
		.VX_bckE_req     (VX_bckE_req),
		.VX_gpr_data     (VX_gpr_datf),
		.VX_exec_unit_req(VX_exec_unit_req_temp),
		.VX_lsu_req      (VX_lsu_req_temp),
		.VX_gpu_inst_req (VX_gpu_inst_req_temp),
		.VX_csr_req      (VX_csr_req_temp)
		);

	wire is_lsu = (|VX_lsu_req_temp.valid);

	wire stall_rest = 0;
	wire flush_rest = schedule_delay;


	wire stall_lsu  = memory_delay;
	wire flush_lsu  = schedule_delay && !stall_lsu;

	wire stall_exec  = exec_delay;
	wire flush_exec  = schedule_delay && !stall_exec;

	wire stall_csr = stall_gpr_csr && VX_bckE_req.is_csr && (|VX_bckE_req.valid);

	assign gpr_stage_delay = stall_lsu || stall_exec || stall_csr;

	`ifdef ASIC
		wire delayed_lsu_last_cycle;

		VX_generic_register #(.N(1)) delayed_reg (
			.clk  (clk),
			.reset(reset),
			.stall(stall_rest),
			.flush(stall_rest),
			.in   (stall_lsu),
			.out  (delayed_lsu_last_cycle)
			);


		wire[`NT_M1:0][31:0] temp_store_data;
		wire[`NT_M1:0][31:0] temp_base_address; // A reg data

		wire[`NT_M1:0][31:0] real_store_data;
		wire[`NT_M1:0][31:0] real_base_address; // A reg data

		wire store_curr_real = !delayed_lsu_last_cycle && stall_lsu;

		VX_generic_register #(.N(`NT*32*2)) lsu_data(
			.clk  (clk),
			.reset(reset),
			.stall(!store_curr_real),
			.flush(stall_rest),
			.in   ({real_store_data, real_base_address}),
			.out  ({temp_store_data, temp_base_address})
			);

		assign real_store_data   = VX_lsu_req_temp.store_data;
		assign real_base_address = VX_lsu_req_temp.base_address;


		assign VX_lsu_req.store_data   = (delayed_lsu_last_cycle) ? temp_store_data   : real_store_data;
		assign VX_lsu_req.base_address = (delayed_lsu_last_cycle) ? temp_base_address : real_base_address;


		VX_generic_register #(.N(77 + `NW_M1 + 1 + (`NT))) lsu_reg(
			.clk  (clk),
			.reset(reset),
			.stall(stall_lsu),
			.flush(flush_lsu),
			.in   ({VX_lsu_req_temp.valid, VX_lsu_req_temp.lsu_pc, VX_lsu_req_temp.warp_num, VX_lsu_req_temp.offset, VX_lsu_req_temp.mem_read, VX_lsu_req_temp.mem_write, VX_lsu_req_temp.rd, VX_lsu_req_temp.wb}),
			.out  ({VX_lsu_req.valid     , VX_lsu_req.lsu_pc     ,VX_lsu_req.warp_num     , VX_lsu_req.offset     , VX_lsu_req.mem_read     , VX_lsu_req.mem_write     , VX_lsu_req.rd     , VX_lsu_req.wb     })
			);

		VX_generic_register #(.N(224 + `NW_M1 + 1 + (`NT))) exec_unit_reg(
			.clk  (clk),
			.reset(reset),
			.stall(stall_exec),
			.flush(flush_exec),
			.in   ({VX_exec_unit_req_temp.valid, VX_exec_unit_req_temp.warp_num, VX_exec_unit_req_temp.curr_PC, VX_exec_unit_req_temp.PC_next, VX_exec_unit_req_temp.rd, VX_exec_unit_req_temp.wb, VX_exec_unit_req_temp.alu_op, VX_exec_unit_req_temp.rs1, VX_exec_unit_req_temp.rs2, VX_exec_unit_req_temp.rs2_src, VX_exec_unit_req_temp.itype_immed, VX_exec_unit_req_temp.upper_immed, VX_exec_unit_req_temp.branch_type, VX_exec_unit_req_temp.jalQual, VX_exec_unit_req_temp.jal, VX_exec_unit_req_temp.jal_offset, VX_exec_unit_req_temp.ebreak, VX_exec_unit_req_temp.wspawn, VX_exec_unit_req_temp.is_csr, VX_exec_unit_req_temp.csr_address, VX_exec_unit_req_temp.csr_immed, VX_exec_unit_req_temp.csr_mask}),
			.out  ({VX_exec_unit_req.valid     , VX_exec_unit_req.warp_num     , VX_exec_unit_req.curr_PC     , VX_exec_unit_req.PC_next     , VX_exec_unit_req.rd     , VX_exec_unit_req.wb     , VX_exec_unit_req.alu_op     , VX_exec_unit_req.rs1     , VX_exec_unit_req.rs2     , VX_exec_unit_req.rs2_src     , VX_exec_unit_req.itype_immed     , VX_exec_unit_req.upper_immed     , VX_exec_unit_req.branch_type     , VX_exec_unit_req.jalQual     , VX_exec_unit_req.jal     , VX_exec_unit_req.jal_offset     , VX_exec_unit_req.ebreak     , VX_exec_unit_req.wspawn     , VX_exec_unit_req.is_csr     , VX_exec_unit_req.csr_address     , VX_exec_unit_req.csr_immed     , VX_exec_unit_req.csr_mask     })
			);

		assign VX_exec_unit_req.a_reg_data = real_base_address;
		assign VX_exec_unit_req.b_reg_data = real_store_data;

		VX_generic_register #(.N(36 + `NW_M1 + 1 + (`NT))) gpu_inst_reg(
			.clk  (clk),
			.reset(reset),
			.stall(stall_rest),
			.flush(flush_rest),
			.in   ({VX_gpu_inst_req_temp.valid, VX_gpu_inst_req_temp.warp_num, VX_gpu_inst_req_temp.is_wspawn, VX_gpu_inst_req_temp.is_tmc, VX_gpu_inst_req_temp.is_split, VX_gpu_inst_req_temp.is_barrier, VX_gpu_inst_req_temp.pc_next}),
			.out  ({VX_gpu_inst_req.valid     , VX_gpu_inst_req.warp_num     , VX_gpu_inst_req.is_wspawn     , VX_gpu_inst_req.is_tmc     , VX_gpu_inst_req.is_split     , VX_gpu_inst_req.is_barrier     , VX_gpu_inst_req.pc_next     })
			);

		assign VX_gpu_inst_req.a_reg_data = real_base_address;
		assign VX_gpu_inst_req.rd2        = real_store_data;

		VX_generic_register #(.N(`NW_M1  + 1 + `NT + 58)) csr_reg(
			.clk  (clk),
			.reset(reset),
			.stall(stall_gpr_csr),
			.flush(flush_rest),
			.in   ({VX_csr_req_temp.valid, VX_csr_req_temp.warp_num, VX_csr_req_temp.rd, VX_csr_req_temp.wb, VX_csr_req_temp.alu_op, VX_csr_req_temp.is_csr, VX_csr_req_temp.csr_address, VX_csr_req_temp.csr_immed, VX_csr_req_temp.csr_mask}),
			.out  ({VX_csr_req.valid     , VX_csr_req.warp_num     , VX_csr_req.rd     , VX_csr_req.wb     , VX_csr_req.alu_op     , VX_csr_req.is_csr     , VX_csr_req.csr_address     , VX_csr_req.csr_immed     , VX_csr_req.csr_mask     })
			);


		// assign 
		
	`else 

    // 341 
	VX_generic_register #(.N(77 + `NW_M1 + 1 + 65*(`NT))) lsu_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_lsu),
		.flush(flush_lsu),
		.in   ({VX_lsu_req_temp.valid, VX_lsu_req_temp.lsu_pc, VX_lsu_req_temp.warp_num, VX_lsu_req_temp.store_data, VX_lsu_req_temp.base_address, VX_lsu_req_temp.offset, VX_lsu_req_temp.mem_read, VX_lsu_req_temp.mem_write, VX_lsu_req_temp.rd, VX_lsu_req_temp.wb}),
		.out  ({VX_lsu_req.valid     , VX_lsu_req.lsu_pc     , VX_lsu_req.warp_num     , VX_lsu_req.store_data     , VX_lsu_req.base_address     , VX_lsu_req.offset     , VX_lsu_req.mem_read     , VX_lsu_req.mem_write     , VX_lsu_req.rd     , VX_lsu_req.wb     })
		);

	VX_generic_register #(.N(224 + `NW_M1 + 1 + 65*(`NT))) exec_unit_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_exec),
		.flush(flush_exec),
		.in   ({VX_exec_unit_req_temp.valid, VX_exec_unit_req_temp.warp_num, VX_exec_unit_req_temp.curr_PC, VX_exec_unit_req_temp.PC_next, VX_exec_unit_req_temp.rd, VX_exec_unit_req_temp.wb, VX_exec_unit_req_temp.a_reg_data, VX_exec_unit_req_temp.b_reg_data, VX_exec_unit_req_temp.alu_op, VX_exec_unit_req_temp.rs1, VX_exec_unit_req_temp.rs2, VX_exec_unit_req_temp.rs2_src, VX_exec_unit_req_temp.itype_immed, VX_exec_unit_req_temp.upper_immed, VX_exec_unit_req_temp.branch_type, VX_exec_unit_req_temp.jalQual, VX_exec_unit_req_temp.jal, VX_exec_unit_req_temp.jal_offset, VX_exec_unit_req_temp.ebreak, VX_exec_unit_req_temp.wspawn, VX_exec_unit_req_temp.is_csr, VX_exec_unit_req_temp.csr_address, VX_exec_unit_req_temp.csr_immed, VX_exec_unit_req_temp.csr_mask}),
		.out  ({VX_exec_unit_req.valid     , VX_exec_unit_req.warp_num     , VX_exec_unit_req.curr_PC     , VX_exec_unit_req.PC_next     , VX_exec_unit_req.rd     , VX_exec_unit_req.wb     , VX_exec_unit_req.a_reg_data     , VX_exec_unit_req.b_reg_data     , VX_exec_unit_req.alu_op     , VX_exec_unit_req.rs1     , VX_exec_unit_req.rs2     , VX_exec_unit_req.rs2_src     , VX_exec_unit_req.itype_immed     , VX_exec_unit_req.upper_immed     , VX_exec_unit_req.branch_type     , VX_exec_unit_req.jalQual     , VX_exec_unit_req.jal     , VX_exec_unit_req.jal_offset     , VX_exec_unit_req.ebreak     , VX_exec_unit_req.wspawn     , VX_exec_unit_req.is_csr     , VX_exec_unit_req.csr_address     , VX_exec_unit_req.csr_immed     , VX_exec_unit_req.csr_mask     })
		);

	VX_generic_register #(.N(68 + `NW_M1 + 1 + 33*(`NT))) gpu_inst_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({VX_gpu_inst_req_temp.valid, VX_gpu_inst_req_temp.warp_num, VX_gpu_inst_req_temp.is_wspawn, VX_gpu_inst_req_temp.is_tmc, VX_gpu_inst_req_temp.is_split, VX_gpu_inst_req_temp.is_barrier, VX_gpu_inst_req_temp.pc_next, VX_gpu_inst_req_temp.a_reg_data, VX_gpu_inst_req_temp.rd2}),
		.out  ({VX_gpu_inst_req.valid     , VX_gpu_inst_req.warp_num     , VX_gpu_inst_req.is_wspawn     , VX_gpu_inst_req.is_tmc     , VX_gpu_inst_req.is_split     , VX_gpu_inst_req.is_barrier     , VX_gpu_inst_req.pc_next     , VX_gpu_inst_req.a_reg_data     , VX_gpu_inst_req.rd2     })
		);

	VX_generic_register #(.N(`NW_M1  + 1 + `NT + 58)) csr_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_gpr_csr),
		.flush(flush_rest),
		.in   ({VX_csr_req_temp.valid, VX_csr_req_temp.warp_num, VX_csr_req_temp.rd, VX_csr_req_temp.wb, VX_csr_req_temp.alu_op, VX_csr_req_temp.is_csr, VX_csr_req_temp.csr_address, VX_csr_req_temp.csr_immed, VX_csr_req_temp.csr_mask}),
		.out  ({VX_csr_req.valid     , VX_csr_req.warp_num     , VX_csr_req.rd     , VX_csr_req.wb     , VX_csr_req.alu_op     , VX_csr_req.is_csr     , VX_csr_req.csr_address     , VX_csr_req.csr_immed     , VX_csr_req.csr_mask     })
		);

	`endif

endmodule : VX_gpr_stage