
`include "VX_define.v"

module VX_fetch (
	input  wire              clk,
	input  wire              reset,
	VX_wstall_inter          VX_wstall,
	VX_join_inter            VX_join,
	input  wire              schedule_delay,
	VX_icache_response_inter icache_response,
	VX_icache_request_inter  icache_request,

	output wire              out_ebreak,
	VX_jal_response_inter    VX_jal_rsp,
	VX_branch_response_inter VX_branch_rsp,
	VX_inst_meta_inter       fe_inst_meta_fd,
	VX_warp_ctl_inter        VX_warp_ctl
);

		// Locals
		wire pipe_stall;


		assign pipe_stall = schedule_delay;

		wire[`NT_M1:0] thread_mask;
		wire[`NW_M1:0] warp_num;
		wire[31:0]     warp_pc;
		VX_warp_scheduler warp_scheduler(
			.clk            (clk),
			.reset          (reset),
			.stall          (pipe_stall),
			// Wspawn
			.wspawn         (VX_warp_ctl.wspawn),
			.wsapwn_pc      (VX_warp_ctl.wspawn_pc),
			// .wspawn_warp_num(VX_warp_ctl.warp_num),
			// CTM
			.ctm            (VX_warp_ctl.change_mask),
			.ctm_mask       (VX_warp_ctl.thread_mask),
			.ctm_warp_num   (VX_warp_ctl.warp_num),
			// WHALT
			.whalt          (VX_warp_ctl.ebreak),
			.whalt_warp_num (VX_warp_ctl.warp_num),
			// Wstall
			.wstall         (VX_wstall.wstall),
			.wstall_warp_num(VX_wstall.warp_num),

			// Join
			.is_join         (VX_join.is_join),
			.join_warp_num   (VX_join.join_warp_num),

			// Split
			.is_split        (VX_warp_ctl.is_split),
			.split_new_mask  (VX_warp_ctl.split_new_mask),
			.split_later_mask(VX_warp_ctl.split_later_mask),
			.split_save_pc   (VX_warp_ctl.split_save_pc),
			.split_warp_num  (VX_warp_ctl.warp_num),

			// JAL
			.jal            (VX_jal_rsp.jal),
			.jal_dest       (VX_jal_rsp.jal_dest),
			.jal_warp_num   (VX_jal_rsp.jal_warp_num),

			// Branch
			.branch_valid   (VX_branch_rsp.valid_branch),
			.branch_dir     (VX_branch_rsp.branch_dir),
			.branch_dest    (VX_branch_rsp.branch_dest),
			.branch_warp_num(VX_branch_rsp.branch_warp_num),

			// Outputs
			.thread_mask    (thread_mask),
			.warp_num       (warp_num),
			.warp_pc        (warp_pc),
			.out_ebreak     (out_ebreak)
			);
	

		assign icache_request.pc_address = warp_pc;
		assign fe_inst_meta_fd.warp_num  = warp_num;
		assign fe_inst_meta_fd.valid     = thread_mask;

		assign fe_inst_meta_fd.instruction = (thread_mask == 0) ? 32'b0 : icache_response.instruction;;
		assign fe_inst_meta_fd.inst_pc     = warp_pc;


endmodule