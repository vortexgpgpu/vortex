`include "VX_define.v"

module VX_front_end (
	input wire clk,
	input wire reset,

	input wire forwarding_fwd_stall,
	input wire memory_delay,


	input wire           execute_branch_stall,

	VX_icache_response_inter icache_response_fe,
	VX_icache_request_inter  icache_request_fe,

	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,

	VX_wb_inter               VX_writeback_inter,
	VX_forward_reqeust_inter  VX_fwd_req_de,
	VX_forward_response_inter VX_fwd_rsp,
	VX_frE_to_bckE_req_inter  VX_bckE_req,


	output wire[11:0] decode_csr_address,
	output wire fetch_delay,
	output wire fetch_ebreak
);

wire[`NW_M1:0] fetch_which_warp;

VX_warp_ctl_inter         VX_warp_ctl();

VX_inst_meta_inter        fe_inst_meta_fd();

VX_frE_to_bckE_req_inter VX_frE_to_bckE_req();
VX_inst_meta_inter       fd_inst_meta_de();

// From decode
wire           decode_branch_stall;
wire           decode_gpr_stall;


wire total_freeze = memory_delay || fetch_delay;


VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.in_memory_delay    (memory_delay),
		.in_branch_stall    (decode_branch_stall),
		.in_fwd_stall       (forwarding_fwd_stall),
		.in_branch_stall_exe(execute_branch_stall),
		.in_gpr_stall     (decode_gpr_stall),
		.VX_jal_rsp         (VX_jal_rsp),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.VX_branch_rsp      (VX_branch_rsp),
		.out_delay          (fetch_delay),
		.out_ebreak         (fetch_ebreak),
		.out_which_wspawn   (fetch_which_warp),
		.fe_inst_meta_fd    (fe_inst_meta_fd)
	);

VX_f_d_reg vx_f_d_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_freeze      (total_freeze),
		.in_gpr_stall (decode_gpr_stall),
		.fe_inst_meta_fd(fe_inst_meta_fd),
		.fd_inst_meta_de(fd_inst_meta_de)
	);


VX_decode vx_decode(
		.clk               (clk),
		.fd_inst_meta_de   (fd_inst_meta_de),
		.VX_writeback_inter(VX_writeback_inter),
		.VX_fwd_rsp        (VX_fwd_rsp),
		.in_which_wspawn   (fetch_which_warp),

		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_fwd_req_de     (VX_fwd_req_de),
		.VX_warp_ctl       (VX_warp_ctl),
		.out_gpr_stall   (decode_gpr_stall),
		.out_branch_stall  (decode_branch_stall)
	);


VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_freeze      (total_freeze),
		.in_gpr_stall (decode_gpr_stall),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);


assign decode_csr_address = VX_frE_to_bckE_req.csr_address;


endmodule


