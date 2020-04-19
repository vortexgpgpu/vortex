`include "VX_define.vh"

module VX_front_end (
	input wire clk,
	input wire reset,

	input wire           schedule_delay,

	VX_warp_ctl_inter        vx_warp_ctl,

	VX_gpu_dcache_rsp_inter  vx_icache_rsp,
	VX_gpu_dcache_req_inter  vx_icache_req,

	VX_jal_response_inter    vx_jal_rsp,
	VX_branch_response_inter vx_branch_rsp,

	VX_frE_to_bckE_req_inter vx_bckE_req,

	output wire fetch_ebreak
);


VX_inst_meta_inter        fe_inst_meta_fi();
VX_inst_meta_inter        fe_inst_meta_fi2();
VX_inst_meta_inter        fe_inst_meta_id();

VX_frE_to_bckE_req_inter  vx_frE_to_bckE_req();
VX_inst_meta_inter        fd_inst_meta_de();

wire total_freeze = schedule_delay;
wire icache_stage_delay;

/* verilator lint_off UNUSED */
// wire real_fetch_ebreak;
/* verilator lint_on UNUSED */

wire vortex_ebreak;
wire terminate_sim;

wire[`NW_BITS-1:0] icache_stage_wid;
wire[`NUM_THREADS-1:0]  icache_stage_valids;

reg old_ebreak; // This should be eventually removed
always @(posedge clk) begin
	if (reset) begin
		old_ebreak <= 0;
	end else begin
		old_ebreak <= old_ebreak || fetch_ebreak;
	end
end

assign fetch_ebreak = vortex_ebreak || terminate_sim || old_ebreak;


VX_wstall_inter          vx_wstall();
VX_join_inter            vx_join();

VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.icache_stage_wid   (icache_stage_wid),
		.icache_stage_valids(icache_stage_valids),
		.vx_wstall          (vx_wstall),
		.vx_join            (vx_join),
		.schedule_delay     (schedule_delay),
		.vx_jal_rsp         (vx_jal_rsp),
		.vx_warp_ctl        (vx_warp_ctl),
		.icache_stage_delay (icache_stage_delay),
		.vx_branch_rsp      (vx_branch_rsp),
		.out_ebreak         (vortex_ebreak), // fetch_ebreak
		.fe_inst_meta_fi    (fe_inst_meta_fi)
	);

wire freeze_fi_reg = total_freeze || icache_stage_delay;




VX_f_d_reg vx_f_i_reg(
		.clk            (clk),
		.reset          (reset),
		.in_freeze      (freeze_fi_reg),
		.fe_inst_meta_fd(fe_inst_meta_fi),
		.fd_inst_meta_de(fe_inst_meta_fi2)
	);

VX_icache_stage vx_icache_stage(
	.clk                (clk),
	.reset              (reset),
	.total_freeze       (total_freeze),
	.icache_stage_delay (icache_stage_delay),
	.icache_stage_valids(icache_stage_valids),
	.icache_stage_wid   (icache_stage_wid),
	.fe_inst_meta_fi    (fe_inst_meta_fi2),
	.fe_inst_meta_id    (fe_inst_meta_id),
	.vx_icache_rsp      (vx_icache_rsp),
	.vx_icache_req      (vx_icache_req)
	);


VX_i_d_reg vx_i_d_reg(
		.clk            (clk),
		.reset          (reset),
		.in_freeze      (total_freeze),
		.fe_inst_meta_fd(fe_inst_meta_id),
		.fd_inst_meta_de(fd_inst_meta_de)
	);


VX_decode vx_decode(
		.fd_inst_meta_de   (fd_inst_meta_de),
		.vx_frE_to_bckE_req(vx_frE_to_bckE_req),
		.vx_wstall         (vx_wstall),
		.vx_join           (vx_join),
		.terminate_sim     (terminate_sim)
	);

wire no_br_stall = 0;

VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_branch_stall(no_br_stall),
		.in_freeze      (total_freeze),
		.vx_frE_to_bckE_req(vx_frE_to_bckE_req),
		.vx_bckE_req       (vx_bckE_req)
	);

endmodule


