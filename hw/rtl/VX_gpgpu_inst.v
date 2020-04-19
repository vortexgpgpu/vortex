`include "VX_define.vh"

module VX_gpgpu_inst (
	// Input
	VX_gpu_inst_req_inter    vx_gpu_inst_req,

	// Output
	VX_warp_ctl_inter        vx_warp_ctl
);
	wire[`NUM_THREADS-1:0] curr_valids = vx_gpu_inst_req.valid;
	wire is_split              = (vx_gpu_inst_req.is_split);

	wire[`NUM_THREADS-1:0] tmc_new_mask;
	wire all_threads = `NUM_THREADS < vx_gpu_inst_req.a_reg_data[0];
	
	genvar curr_t;
	generate
	for (curr_t = 0; curr_t < `NUM_THREADS; curr_t=curr_t+1) begin : tmc_new_mask_init
		assign tmc_new_mask[curr_t] = all_threads ? 1 : curr_t < vx_gpu_inst_req.a_reg_data[0];
	end
	endgenerate

	wire valid_inst = (|curr_valids);

	assign vx_warp_ctl.warp_num    = vx_gpu_inst_req.warp_num;
	assign vx_warp_ctl.change_mask = (vx_gpu_inst_req.is_tmc) && valid_inst;
	assign vx_warp_ctl.thread_mask = vx_gpu_inst_req.is_tmc ? tmc_new_mask : 0;

	// assign vx_warp_ctl.ebreak = (vx_gpu_inst_req.a_reg_data[0] == 0) && valid_inst;
	assign vx_warp_ctl.ebreak = vx_warp_ctl.change_mask && (vx_warp_ctl.thread_mask == 0);

	wire       wspawn     = vx_gpu_inst_req.is_wspawn;
	wire[31:0] wspawn_pc  = vx_gpu_inst_req.rd2;
	wire       all_active = `NUM_WARPS < vx_gpu_inst_req.a_reg_data[0];
	wire[`NUM_WARPS-1:0] wspawn_new_active;

	genvar curr_w;
	generate
	for (curr_w = 0; curr_w < `NUM_WARPS; curr_w=curr_w+1) begin : wspawn_new_active_init
		assign wspawn_new_active[curr_w] = all_active ? 1 : curr_w < vx_gpu_inst_req.a_reg_data[0];
	end
	endgenerate

	assign vx_warp_ctl.is_barrier = vx_gpu_inst_req.is_barrier && valid_inst;
	assign vx_warp_ctl.barrier_id = vx_gpu_inst_req.a_reg_data[0];

/* verilator lint_off UNUSED */
	wire[31:0] num_warps_m1       = vx_gpu_inst_req.rd2 - 1;
/* verilator lint_on UNUSED */

	assign vx_warp_ctl.num_warps  = num_warps_m1[$clog2(`NUM_WARPS):0];

	assign vx_warp_ctl.wspawn            = wspawn;
	assign vx_warp_ctl.wspawn_pc         = wspawn_pc;
	assign vx_warp_ctl.wspawn_new_active = wspawn_new_active;

	wire[`NUM_THREADS-1:0] split_new_use_mask;
	wire[`NUM_THREADS-1:0] split_new_later_mask;

	// VX_gpu_inst_req.pc
	genvar curr_s_t;
	generate
	for (curr_s_t = 0; curr_s_t < `NUM_THREADS; curr_s_t=curr_s_t+1) begin : masks_init
		wire curr_bool = (vx_gpu_inst_req.a_reg_data[curr_s_t] == 32'b1);

		assign split_new_use_mask[curr_s_t]   = curr_valids[curr_s_t] & (curr_bool);
		assign split_new_later_mask[curr_s_t] = curr_valids[curr_s_t] & (!curr_bool);
	end
	endgenerate

	wire[$clog2(`NUM_THREADS):0] num_valids;

	VX_countones #(
		.N(`NUM_THREADS)
	) valids_counter (
		.valids(curr_valids),
		.count (num_valids)
	);

	// wire[`NW_BITS-1:0] num_valids = $countones(curr_valids);
	
	assign vx_warp_ctl.is_split         = is_split && (num_valids > 1);
	assign vx_warp_ctl.dont_split       = vx_warp_ctl.is_split && ((split_new_use_mask == 0) || (split_new_use_mask == {`NUM_THREADS{1'b1}}));
	assign vx_warp_ctl.split_new_mask   = split_new_use_mask;
	assign vx_warp_ctl.split_later_mask = split_new_later_mask;
	assign vx_warp_ctl.split_save_pc    = vx_gpu_inst_req.pc_next;
	assign vx_warp_ctl.split_warp_num   = vx_gpu_inst_req.warp_num;

	// vx_gpu_inst_req.is_wspawn
	// vx_gpu_inst_req.is_split
	// vx_gpu_inst_req.is_barrier

endmodule