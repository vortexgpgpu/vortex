module VX_gpr_stage (
	input wire                 clk,
	input wire                 in_fwd_stall,
	// inputs
		// Instruction Information
	VX_frE_to_bckE_req_inter   VX_bckE_req,
		// WriteBack inputs
	VX_wb_inter                VX_writeback_inter,
		// FORWARDING INPUTS
	VX_forward_response_inter  VX_fwd_rsp,




	// Outputs
		// Fwd Request
	VX_forward_reqeust_inter   VX_fwd_req_de,
		// Warp Control
	VX_warp_ctl_inter          VX_warp_ctl,
		// Original Request 1 cycle later
	VX_frE_to_bckE_req_inter   VX_bckE_req_out,
		// Data Read
	VX_gpr_data_inter          VX_gpr_data,

	output wire                out_gpr_stall
);


	wire[31:0] curr_PC = VX_bckE_req.curr_PC;
	wire[2:0] branchType = VX_bckE_req.branch_type;

	wire jalQual = VX_bckE_req.jalQual;


	assign VX_fwd_req_de.src1        = VX_bckE_req.rs1;
	assign VX_fwd_req_de.src2        = VX_bckE_req.rs2;
	assign VX_fwd_req_de.warp_num    = VX_bckE_req.warp_num;

	VX_gpr_read_inter VX_gpr_read();
	assign VX_gpr_read.rs1      = VX_bckE_req.rs1;
	assign VX_gpr_read.rs2      = VX_bckE_req.rs2;
	assign VX_gpr_read.warp_num = VX_bckE_req.warp_num;

	VX_gpr_jal_inter VX_gpr_jal();
	assign VX_gpr_jal.is_jal  = VX_bckE_req.jalQual;
	assign VX_gpr_jal.curr_PC = VX_bckE_req.curr_PC;


	VX_gpr_wrapper vx_grp_wrapper(
			.clk            (clk),
			.VX_writeback_inter(VX_writeback_inter),
			.VX_fwd_rsp        (VX_fwd_rsp),
			.VX_gpr_read       (VX_gpr_read),
			.VX_gpr_jal        (VX_gpr_jal),

			.out_a_reg_data (VX_gpr_datf.a_reg_data),
			.out_b_reg_data (VX_gpr_datf.b_reg_data),
			.out_gpr_stall(out_gpr_stall)
		);

	// assign VX_bckE_req.is_csr   = is_csr;
	// assign VX_bckE_req_out.csr_mask = (VX_bckE_req.sr_immed == 1'b1) ?  {27'h0, VX_bckE_req.rs1} : VX_gpr_data.a_reg_data[0];

	VX_gpr_data_inter           VX_gpr_datf;
	VX_generic_register #(.N(256)) d_e_reg 
	(
		.clk  (clk),
		.reset(0),
		.stall(0),
		.flush(0),
		.in   ({VX_gpr_datf.a_reg_data, VX_gpr_datf.b_reg_data}),
		.out  ({VX_gpr_data.a_reg_data, VX_gpr_data.b_reg_data})
	);

	VX_d_e_reg vx_d_e_reg(
			.clk               (clk),
			.reset             (0),
			.in_fwd_stall      (in_fwd_stall),
			.in_branch_stall   (0),
			.in_freeze         (0),
			.in_gpr_stall      (out_gpr_stall),
			.VX_frE_to_bckE_req(VX_bckE_req),
			.VX_bckE_req       (VX_bckE_req_out)
		);


	// assign VX_warp_ctl.warp_num    = VX_bckE_req_out.warp_num;
	// assign VX_warp_ctl.wspawn      = VX_bckE_req_out.wspawn;
	// assign VX_warp_ctl.wspawn_pc   = VX_bckE_req_out.a_reg_data[0];

	// assign VX_warp_ctl.thread_mask = is_jalrs ? jalrs_thread_mask : jmprt_thread_mask;
	// assign VX_warp_ctl.change_mask = is_jalrs || is_jmprt;
	// assign VX_warp_ctl.ebreak      = VX_bckE_req_out.ebreak;


	assign VX_warp_ctl.warp_num    = 0;
	assign VX_warp_ctl.wspawn      = 0;
	assign VX_warp_ctl.wspawn_pc   = 0;

	assign VX_warp_ctl.thread_mask = 0;
	assign VX_warp_ctl.change_mask = 0;
	assign VX_warp_ctl.ebreak      = 0;

endmodule