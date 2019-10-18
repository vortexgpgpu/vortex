module VX_back_end (
	input wire clk, 
	input wire reset, 
	input wire schedule_delay,

	input wire[31:0]          csr_decode_csr_data,
	output wire               out_mem_delay,

	VX_jal_response_inter     VX_jal_rsp,
	VX_branch_response_inter  VX_branch_rsp,


	VX_frE_to_bckE_req_inter  VX_bckE_req,
	VX_wb_inter               VX_writeback_inter,

	VX_warp_ctl_inter         VX_warp_ctl,

	VX_dcache_response_inter  VX_dcache_rsp,
	VX_dcache_request_inter   VX_dcache_req,


	VX_csr_write_request_inter VX_csr_w_req
);


wire[11:0]      execute_csr_address;
wire            execute_is_csr;
reg[31:0]       execute_csr_result;
wire            execute_jal;
wire[31:0]      execute_jal_dest;





VX_mw_wb_inter           VX_mw_wb();


VX_mem_req_inter  VX_exe_mem_req();
VX_mem_req_inter  VX_mem_req();


VX_gpr_data_inter           VX_gpr_data();

VX_frE_to_bckE_req_inter VX_bckE_req_out();

// LSU input + output
VX_lsu_req_inter         VX_lsu_req();
VX_inst_mem_wb_inter     VX_mem_wb();

// Exec unit input + output
VX_exec_unit_req_inter   VX_exec_unit_req();
VX_inst_exec_wb_inter    VX_inst_exec_wb();

VX_gpr_stage VX_gpr_stage(
	.clk               (clk),
	.schedule_delay    (schedule_delay),
	.VX_writeback_inter(VX_writeback_inter),
	.VX_bckE_req       (VX_bckE_req),
	.VX_warp_ctl       (VX_warp_ctl),
	.VX_bckE_req_out   (VX_bckE_req_out),
	.VX_gpr_data       (VX_gpr_data)
	);


VX_inst_multiplex VX_inst_mult(
	.VX_bckE_req     (VX_bckE_req_out),
	.VX_gpr_data     (VX_gpr_data),
	.VX_exec_unit_req(VX_exec_unit_req),
	.VX_lsu_req      (VX_lsu_req)
	);


VX_lsu load_store_unit(
	// .clk          (clk),
	.VX_lsu_req   (VX_lsu_req),
	.VX_mem_wb    (VX_mem_wb),
	.VX_dcache_rsp(VX_dcache_rsp),
	.VX_dcache_req(VX_dcache_req),
	.out_delay    (out_mem_delay)
	);


VX_execute_unit VX_execUnit(
	// .clk             (clk),
	.VX_exec_unit_req(VX_exec_unit_req),
	.VX_inst_exec_wb (VX_inst_exec_wb),
	.VX_jal_rsp      (VX_jal_rsp),
	.VX_branch_rsp   (VX_branch_rsp),

	.in_csr_data     (csr_decode_csr_data),
	.out_csr_address (VX_csr_w_req.csr_address),
	.out_is_csr      (VX_csr_w_req.is_csr),
	.out_csr_result  (VX_csr_w_req.csr_result)
	);

VX_writeback VX_wb(
	.VX_mem_wb         (VX_mem_wb),
	.VX_inst_exec_wb   (VX_inst_exec_wb),

	.VX_writeback_inter(VX_writeback_inter)
	);

endmodule