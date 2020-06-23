`include "VX_define.v"

module VX_front_end (
	input wire clk,
	input wire reset,

	input wire           schedule_delay,

	VX_warp_ctl_inter         VX_warp_ctl,

	VX_icache_response_inter icache_response_fe,
	VX_icache_request_inter  icache_request_fe,

	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,

	VX_frE_to_bckE_req_inter  VX_bckE_req,

	output wire fetch_ebreak
);


VX_inst_meta_inter        fe_inst_meta_fd();

VX_frE_to_bckE_req_inter VX_frE_to_bckE_req();
VX_inst_meta_inter       fd_inst_meta_de();

wire total_freeze = schedule_delay;

/* verilator lint_off UNUSED */
// wire real_fetch_ebreak;
/* verilator lint_on UNUSED */

wire vortex_ebreak;
wire terminate_sim;

assign fetch_ebreak = vortex_ebreak || terminate_sim;


VX_wstall_inter          VX_wstall();
VX_join_inter            VX_join();

VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.VX_wstall          (VX_wstall),
		.VX_join            (VX_join),
		.schedule_delay     (schedule_delay),
		.VX_jal_rsp         (VX_jal_rsp),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.VX_branch_rsp      (VX_branch_rsp),
		.out_ebreak         (vortex_ebreak), // fetch_ebreak
		.fe_inst_meta_fd    (fe_inst_meta_fd)
	);

VX_f_d_reg vx_f_d_reg(
		.clk            (clk),
		.reset          (reset),
		.in_freeze      (total_freeze),
		.fe_inst_meta_fd(fe_inst_meta_fd),
		.fd_inst_meta_de(fd_inst_meta_de)
	);


VX_decode vx_decode(
		.fd_inst_meta_de   (fd_inst_meta_de),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_wstall         (VX_wstall),
		.VX_join           (VX_join),
		.terminate_sim     (terminate_sim)
	);

wire no_br_stall = 0;

VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_branch_stall(no_br_stall),
		.in_freeze      (total_freeze),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);

endmodule


