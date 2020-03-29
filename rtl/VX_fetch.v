`include "VX_define.v"

module VX_fetch (
	input  wire              clk,
	input  wire              reset,
	VX_wstall_inter          VX_wstall,
	VX_join_inter            VX_join,
	input  wire              schedule_delay,
	input  wire              icache_stage_delay,
	input  wire[`NW_M1:0]    icache_stage_wid,
	input  wire[`NT-1:0]     icache_stage_valids,

	output wire              out_ebreak,
	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,
	VX_inst_meta_inter       fe_inst_meta_fi,
	VX_warp_ctl_inter        VX_warp_ctl
);

		wire[`NT_M1:0] thread_mask;
		wire[`NW_M1:0] warp_num;
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

			.is_barrier       (VX_warp_ctl.is_barrier),
			.barrier_id       (VX_warp_ctl.barrier_id),
			.num_warps        (VX_warp_ctl.num_warps),
			.barrier_warp_num (VX_warp_ctl.warp_num),

			// Wspawn
			.wspawn           (VX_warp_ctl.wspawn),
			.wsapwn_pc        (VX_warp_ctl.wspawn_pc),
			.wspawn_new_active(VX_warp_ctl.wspawn_new_active),
			// CTM
			.ctm              (VX_warp_ctl.change_mask),
			.ctm_mask         (VX_warp_ctl.thread_mask),
			.ctm_warp_num     (VX_warp_ctl.warp_num),
			// WHALT
			.whalt            (VX_warp_ctl.ebreak),
			.whalt_warp_num   (VX_warp_ctl.warp_num),
			// Wstall
			.wstall           (VX_wstall.wstall),
			.wstall_warp_num  (VX_wstall.warp_num),

			// Lock/release Stuff
			.icache_stage_valids(icache_stage_valids),
			.icache_stage_wid   (icache_stage_wid),

			// Join
			.is_join           (VX_join.is_join),
			.join_warp_num     (VX_join.join_warp_num),

			// Split
			.is_split          (VX_warp_ctl.is_split),
			.dont_split        (VX_warp_ctl.dont_split),
			.split_new_mask    (VX_warp_ctl.split_new_mask),
			.split_later_mask  (VX_warp_ctl.split_later_mask),
			.split_save_pc     (VX_warp_ctl.split_save_pc),
			.split_warp_num    (VX_warp_ctl.warp_num),

			// JAL
			.jal              (VX_jal_rsp.jal),
			.jal_dest         (VX_jal_rsp.jal_dest),
			.jal_warp_num     (VX_jal_rsp.jal_warp_num),

			// Branch
			.branch_valid     (VX_branch_rsp.valid_branch),
			.branch_dir       (VX_branch_rsp.branch_dir),
			.branch_dest      (VX_branch_rsp.branch_dest),
			.branch_warp_num  (VX_branch_rsp.branch_warp_num),

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

		wire start_mat_add = scheduled_warp && (warp_pc == 32'h80000ed8) && (warp_num == 0);
		wire end_mat_add   = scheduled_warp && (warp_pc == 32'h80000fbc) && (warp_num == 0);


endmodule