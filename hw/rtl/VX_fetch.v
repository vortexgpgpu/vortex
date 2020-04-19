`include "VX_define.vh"

module VX_fetch (
	input  wire              clk,
	input  wire              reset,
	VX_wstall_inter          vx_wstall,
	VX_join_inter            vx_join,
	input  wire              schedule_delay,
	input  wire              icache_stage_delay,
	input  wire[`NW_BITS-1:0]    icache_stage_wid,
	input  wire[`NUM_THREADS-1:0]     icache_stage_valids,

	output wire              out_ebreak,
	VX_jal_response_inter    vx_jal_rsp,
	VX_branch_response_inter vx_branch_rsp,
	VX_inst_meta_inter       fe_inst_meta_fi,
	VX_warp_ctl_inter        vx_warp_ctl
);

	wire[`NUM_THREADS-1:0] thread_mask;
	wire[`NW_BITS-1:0] warp_num;
	wire[31:0]     warp_pc;
	wire           scheduled_warp;


	wire pipe_stall;


	// Only reason this is there is because there is a hidden assumption that decode is exactly after fetch

	// Locals


	assign pipe_stall = schedule_delay || icache_stage_delay;

	VX_warp_scheduler warp_scheduler(
		.clk              (clk),
		.reset            (reset),
		.stall            (pipe_stall),

		.is_barrier       (vx_warp_ctl.is_barrier),
		.barrier_id       (vx_warp_ctl.barrier_id),
		.num_warps        (vx_warp_ctl.num_warps),
		.barrier_warp_num (vx_warp_ctl.warp_num),

		// Wspawn
		.wspawn           (vx_warp_ctl.wspawn),
		.wsapwn_pc        (vx_warp_ctl.wspawn_pc),
		.wspawn_new_active(vx_warp_ctl.wspawn_new_active),
		// CTM
		.ctm              (vx_warp_ctl.change_mask),
		.ctm_mask         (vx_warp_ctl.thread_mask),
		.ctm_warp_num     (vx_warp_ctl.warp_num),
		// WHALT
		.whalt            (vx_warp_ctl.ebreak),
		.whalt_warp_num   (vx_warp_ctl.warp_num),
		// Wstall
		.wstall           (vx_wstall.wstall),
		.wstall_warp_num  (vx_wstall.warp_num),

		// Lock/release Stuff
		.icache_stage_valids(icache_stage_valids),
		.icache_stage_wid   (icache_stage_wid),

		// Join
		.is_join           (vx_join.is_join),
		.join_warp_num     (vx_join.join_warp_num),

		// Split
		.is_split          (vx_warp_ctl.is_split),
		.dont_split        (vx_warp_ctl.dont_split),
		.split_new_mask    (vx_warp_ctl.split_new_mask),
		.split_later_mask  (vx_warp_ctl.split_later_mask),
		.split_save_pc     (vx_warp_ctl.split_save_pc),
		.split_warp_num    (vx_warp_ctl.warp_num),

		// JAL
		.jal              (vx_jal_rsp.jal),
		.jal_dest         (vx_jal_rsp.jal_dest),
		.jal_warp_num     (vx_jal_rsp.jal_warp_num),

		// Branch
		.branch_valid     (vx_branch_rsp.valid_branch),
		.branch_dir       (vx_branch_rsp.branch_dir),
		.branch_dest      (vx_branch_rsp.branch_dest),
		.branch_warp_num  (vx_branch_rsp.branch_warp_num),

		// Outputs
		.thread_mask      (thread_mask),
		.warp_num         (warp_num),
		.warp_pc          (warp_pc),
		.out_ebreak       (out_ebreak),
		.scheduled_warp   (scheduled_warp)
		);

	assign fe_inst_meta_fi.warp_num    = warp_num;
	assign fe_inst_meta_fi.valid       = thread_mask;
	assign fe_inst_meta_fi.instruction = 32'h0;
	assign fe_inst_meta_fi.inst_pc     = warp_pc;
`DEBUG_BEGIN
	wire start_mat_add = scheduled_warp && (warp_pc == 32'h80000ed8) && (warp_num == 0);
	wire end_mat_add   = scheduled_warp && (warp_pc == 32'h80000fbc) && (warp_num == 0);
`DEBUG_END

endmodule