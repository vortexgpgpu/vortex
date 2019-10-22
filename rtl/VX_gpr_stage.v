
`include "VX_define.v"

module VX_gpr_stage (
	input wire                 clk,
	input wire                 reset,
	input wire                 schedule_delay,

	input  wire                memory_delay,
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

	VX_gpr_jal_inter VX_gpr_jal();
	assign VX_gpr_jal.is_jal  = VX_bckE_req.jalQual;
	assign VX_gpr_jal.curr_PC = VX_bckE_req.curr_PC;


	VX_gpr_data_inter           VX_gpr_datf();


	VX_gpr_wrapper vx_grp_wrapper(
			.clk            (clk),
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


	assign gpr_stage_delay = stall_lsu;


	VX_generic_register #(.N(308)) lsu_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_lsu),
		.flush(flush_lsu),
		.in   ({VX_lsu_req_temp.valid, VX_lsu_req_temp.warp_num, VX_lsu_req_temp.store_data, VX_lsu_req_temp.base_address, VX_lsu_req_temp.offset, VX_lsu_req_temp.mem_read, VX_lsu_req_temp.mem_write, VX_lsu_req_temp.rd, VX_lsu_req_temp.wb}),
		.out  ({VX_lsu_req.valid     , VX_lsu_req.warp_num     , VX_lsu_req.store_data     , VX_lsu_req.base_address     , VX_lsu_req.offset     , VX_lsu_req.mem_read     , VX_lsu_req.mem_write     , VX_lsu_req.rd     , VX_lsu_req.wb     })
		);

	VX_generic_register #(.N(487)) exec_unit_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({VX_exec_unit_req_temp.valid, VX_exec_unit_req_temp.warp_num, VX_exec_unit_req_temp.curr_PC, VX_exec_unit_req_temp.PC_next, VX_exec_unit_req_temp.rd, VX_exec_unit_req_temp.wb, VX_exec_unit_req_temp.a_reg_data, VX_exec_unit_req_temp.b_reg_data, VX_exec_unit_req_temp.alu_op, VX_exec_unit_req_temp.rs1, VX_exec_unit_req_temp.rs2, VX_exec_unit_req_temp.rs2_src, VX_exec_unit_req_temp.itype_immed, VX_exec_unit_req_temp.upper_immed, VX_exec_unit_req_temp.branch_type, VX_exec_unit_req_temp.jalQual, VX_exec_unit_req_temp.jal, VX_exec_unit_req_temp.jal_offset, VX_exec_unit_req_temp.ebreak, VX_exec_unit_req_temp.wspawn, VX_exec_unit_req_temp.is_csr, VX_exec_unit_req_temp.csr_address, VX_exec_unit_req_temp.csr_immed, VX_exec_unit_req_temp.csr_mask}),
		.out  ({VX_exec_unit_req.valid     , VX_exec_unit_req.warp_num     , VX_exec_unit_req.curr_PC     , VX_exec_unit_req.PC_next     , VX_exec_unit_req.rd     , VX_exec_unit_req.wb     , VX_exec_unit_req.a_reg_data     , VX_exec_unit_req.b_reg_data     , VX_exec_unit_req.alu_op     , VX_exec_unit_req.rs1     , VX_exec_unit_req.rs2     , VX_exec_unit_req.rs2_src     , VX_exec_unit_req.itype_immed     , VX_exec_unit_req.upper_immed     , VX_exec_unit_req.branch_type     , VX_exec_unit_req.jalQual     , VX_exec_unit_req.jal     , VX_exec_unit_req.jal_offset     , VX_exec_unit_req.ebreak     , VX_exec_unit_req.wspawn     , VX_exec_unit_req.is_csr     , VX_exec_unit_req.csr_address     , VX_exec_unit_req.csr_immed     , VX_exec_unit_req.csr_mask     })
		);

	VX_generic_register #(.N(203)) gpu_inst_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({VX_gpu_inst_req_temp.valid, VX_gpu_inst_req_temp.warp_num, VX_gpu_inst_req_temp.is_wspawn, VX_gpu_inst_req_temp.is_tmc, VX_gpu_inst_req_temp.is_split, VX_gpu_inst_req_temp.is_barrier, VX_gpu_inst_req_temp.pc_next, VX_gpu_inst_req_temp.a_reg_data, VX_gpu_inst_req_temp.rd2}),
		.out  ({VX_gpu_inst_req.valid     , VX_gpu_inst_req.warp_num     , VX_gpu_inst_req.is_wspawn     , VX_gpu_inst_req.is_tmc     , VX_gpu_inst_req.is_split     , VX_gpu_inst_req.is_barrier     , VX_gpu_inst_req.pc_next     , VX_gpu_inst_req.a_reg_data     , VX_gpu_inst_req.rd2     })
		);

	VX_generic_register #(.N(60)) csr_reg(
		.clk  (clk),
		.reset(reset),
		.stall(stall_rest),
		.flush(flush_rest),
		.in   ({VX_csr_req_temp.valid, VX_csr_req_temp.warp_num, VX_csr_req_temp.rd, VX_csr_req_temp.wb, VX_csr_req_temp.is_csr, VX_csr_req_temp.csr_address, VX_csr_req_temp.csr_immed, VX_csr_req_temp.csr_mask}),
		.out  ({VX_csr_req.valid     , VX_csr_req.warp_num     , VX_csr_req.rd     , VX_csr_req.wb     , VX_csr_req.is_csr     , VX_csr_req.csr_address     , VX_csr_req.csr_immed     , VX_csr_req.csr_mask     })
		);


	// wire zero_temp = 0;

	// VX_generic_register #(.N(256)) reg_data 
	// (
	// 	.clk  (clk),
	// 	.reset(reset),
	// 	.stall(zero_temp),
	// 	.flush(zero_temp),
	// 	.in   ({VX_gpr_datf.a_reg_data, VX_gpr_datf.b_reg_data}),
	// 	.out  ({VX_gpr_data.a_reg_data, VX_gpr_data.b_reg_data})
	// );

	// wire stall = schedule_delay;


	// VX_d_e_reg gpr_stage_reg(
	// 		.clk               (clk),
	// 		.reset             (reset),
	// 		.in_branch_stall   (stall),
	// 		.in_freeze         (zero_temp),
	// 		.VX_frE_to_bckE_req(VX_bckE_req),
	// 		.VX_bckE_req       (VX_bckE_req_out)
	// 	);

endmodule