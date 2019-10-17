`include "VX_define.v"

module VX_front_end (
	input wire clk,
	input wire reset,

	input wire forwarding_fwd_stall,
	input wire memory_delay,

	input wire           execute_branch_stall,
	input wire           in_gpr_stall,
	input wire           schedule_delay,

	VX_warp_ctl_inter         VX_warp_ctl,

	VX_icache_response_inter icache_response_fe,
	VX_icache_request_inter  icache_request_fe,

	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,

	VX_frE_to_bckE_req_inter  VX_bckE_req,


	output wire[11:0] decode_csr_address,
	output wire fetch_delay,
	output wire fetch_ebreak
);


VX_inst_meta_inter        fe_inst_meta_fd();

VX_frE_to_bckE_req_inter VX_frE_to_bckE_req();
VX_inst_meta_inter       fd_inst_meta_de();

// From decode
wire           decode_branch_stall;
wire           decode_gpr_stall;


wire total_freeze = memory_delay || fetch_delay || in_gpr_stall || schedule_delay;

/* verilator lint_off UNUSED */
wire real_fetch_ebreak;
/* verilator lint_on UNUSED */

VX_fetch vx_fetch(
		.clk                (clk),
		.in_memory_delay    (memory_delay),
		.in_branch_stall    (decode_branch_stall),
		.in_fwd_stall       (forwarding_fwd_stall),
		.schedule_delay     (schedule_delay),
		.in_branch_stall_exe(execute_branch_stall),
		.in_gpr_stall     (decode_gpr_stall),
		.VX_jal_rsp         (VX_jal_rsp),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.VX_branch_rsp      (VX_branch_rsp),
		.out_delay          (fetch_delay),
		.out_ebreak         (real_fetch_ebreak), // fetch_ebreak
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
		.fd_inst_meta_de   (fd_inst_meta_de),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.out_gpr_stall     (decode_gpr_stall),
		.out_branch_stall  (decode_branch_stall),
		.out_ebreak        (fetch_ebreak)
	);

wire special_what = total_freeze || forwarding_fwd_stall;

wire temp_fwd_stall = 0;

VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (temp_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_freeze      (special_what),
		.in_gpr_stall (decode_gpr_stall),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);


assign decode_csr_address = VX_frE_to_bckE_req.csr_address;


endmodule


