module VX_gpr_stage (
	input wire                 clk,
	input wire                 schedule_delay,
	// inputs
		// Instruction Information
	VX_frE_to_bckE_req_inter   VX_bckE_req,
		// WriteBack inputs
	VX_wb_inter                VX_writeback_inter,




	// Outputs
		// Original Request 1 cycle later
	VX_frE_to_bckE_req_inter   VX_bckE_req_out,
		// Data Read
	VX_gpr_data_inter          VX_gpr_data
);


	wire[31:0] curr_PC = VX_bckE_req.curr_PC;
	wire[2:0] branchType = VX_bckE_req.branch_type;

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

	wire zero_temp = 0;

	VX_generic_register #(.N(256)) reg_data 
	(
		.clk  (clk),
		.reset(zero_temp),
		.stall(zero_temp),
		.flush(zero_temp),
		.in   ({VX_gpr_datf.a_reg_data, VX_gpr_datf.b_reg_data}),
		.out  ({VX_gpr_data.a_reg_data, VX_gpr_data.b_reg_data})
	);

	wire stall = schedule_delay;


	VX_d_e_reg gpr_stage_reg(
			.clk               (clk),
			.reset             (zero_temp),
			.in_branch_stall   (stall),
			.in_freeze         (zero_temp),
			.VX_frE_to_bckE_req(VX_bckE_req),
			.VX_bckE_req       (VX_bckE_req_out)
		);

endmodule