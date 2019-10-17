module VX_back_end (
	input wire clk, 
	input wire reset, 
	input wire fetch_delay,
	input wire schedule_delay,

	input wire[31:0]         csr_decode_csr_data,
	output wire              execute_branch_stall,
	input wire               in_fwd_stall,

	output wire              out_mem_delay,
	output wire              out_gpr_stall,

	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,


	VX_frE_to_bckE_req_inter VX_bckE_req,
	VX_wb_inter              VX_writeback_inter,

	VX_warp_ctl_inter        VX_warp_ctl,

	VX_dcache_response_inter VX_dcache_rsp,
	VX_dcache_request_inter VX_dcache_req,

	VX_forward_reqeust_inter  VX_fwd_req_de,
	VX_forward_response_inter VX_fwd_rsp,

	VX_forward_exe_inter     VX_fwd_exe,
	VX_forward_mem_inter     VX_fwd_mem,
	VX_forward_wb_inter      VX_fwd_wb,


	VX_csr_write_request_inter VX_csr_w_req
);

wire   memory_delay;

assign out_mem_delay = memory_delay;


wire total_freeze = fetch_delay || memory_delay;

wire[11:0]      execute_csr_address;
wire            execute_is_csr;
reg[31:0]       execute_csr_result;
wire            execute_jal;
wire[31:0]      execute_jal_dest;





VX_mw_wb_inter           VX_mw_wb();
VX_inst_mem_wb_inter     VX_mem_wb();


VX_mem_req_inter  VX_exe_mem_req();
VX_mem_req_inter  VX_mem_req();


VX_gpr_data_inter           VX_gpr_data();

VX_frE_to_bckE_req_inter VX_bckE_req_out();

VX_gpr_stage VX_gpr_stage(
	.clk               (clk),
	.schedule_delay    (schedule_delay),
	.VX_writeback_inter(VX_writeback_inter),
	.VX_fwd_rsp        (VX_fwd_rsp),
	.in_fwd_stall      (in_fwd_stall),
	.VX_bckE_req       (VX_bckE_req),
	.VX_warp_ctl       (VX_warp_ctl),
	.VX_bckE_req_out   (VX_bckE_req_out),
	.VX_gpr_data       (VX_gpr_data),
	.VX_fwd_req_de     (VX_fwd_req_de),
	.out_gpr_stall     (out_gpr_stall)
	);


VX_execute vx_execute(
		.VX_bckE_req      (VX_bckE_req_out),
		.VX_gpr_data      (VX_gpr_data),
		.VX_fwd_exe       (VX_fwd_exe),
		.in_csr_data      (csr_decode_csr_data),

		.VX_exe_mem_req   (VX_exe_mem_req),
		.out_csr_address  (execute_csr_address),
		.out_is_csr       (execute_is_csr),
		.out_csr_result   (execute_csr_result),
		.out_jal          (execute_jal),
		.out_jal_dest     (execute_jal_dest),
		.out_branch_stall (execute_branch_stall)
	);


assign VX_jal_rsp.jal_warp_num = VX_mem_req.warp_num;

VX_e_m_reg vx_e_m_reg(
		.clk              (clk),
		.reset            (reset),
		.in_csr_address   (execute_csr_address),
		.in_is_csr        (execute_is_csr),
		.in_csr_result    (execute_csr_result),
		.in_jal           (execute_jal),
		.in_jal_dest      (execute_jal_dest),
		.in_freeze        (total_freeze),
		.VX_exe_mem_req   (VX_exe_mem_req),

		.VX_mem_req       (VX_mem_req),
		.out_csr_address  (VX_csr_w_req.csr_address),
		.out_is_csr       (VX_csr_w_req.is_csr),
		.out_csr_result   (VX_csr_w_req.csr_result),
		.out_jal          (VX_jal_rsp.jal),
		.out_jal_dest     (VX_jal_rsp.jal_dest)
	);

VX_memory vx_memory(
		.VX_mem_req                   (VX_mem_req),
		.VX_mem_wb                    (VX_mem_wb),
		.VX_fwd_mem                   (VX_fwd_mem),
		.out_delay                    (memory_delay),

		.VX_branch_rsp                (VX_branch_rsp),

		.VX_dcache_rsp(VX_dcache_rsp),
		.VX_dcache_req (VX_dcache_req)
	);

// VX_m_w_reg vx_m_w_reg(
// 		.clk       (clk),
// 		.reset     (reset),
// 		.in_freeze (total_freeze),
// 		.VX_mem_wb (VX_mem_wb),
// 		.VX_mw_wb  (VX_mw_wb)
// 	);

assign VX_mw_wb.alu_result = VX_mem_wb.alu_result;
assign VX_mw_wb.mem_result = VX_mem_wb.mem_result;
assign VX_mw_wb.rd         = VX_mem_wb.rd;
assign VX_mw_wb.wb         = VX_mem_wb.wb;
assign VX_mw_wb.PC_next    = VX_mem_wb.PC_next;
assign VX_mw_wb.valid      = VX_mem_wb.valid;
assign VX_mw_wb.warp_num   = VX_mem_wb.warp_num;


VX_writeback vx_writeback(
		.VX_mw_wb          (VX_mw_wb),
		.VX_fwd_wb         (VX_fwd_wb),
		.VX_writeback_inter(VX_writeback_inter)
	);

endmodule