`include "VX_define.v"

module VX_gpgpu_inst (
	// Input
	VX_gpu_inst_req_inter    VX_gpu_inst_req,

	// Output
	VX_warp_ctl_inter        VX_warp_ctl
);


	wire[`NT_M1:0] curr_valids = VX_gpu_inst_req.valid;
	wire is_split              = (VX_gpu_inst_req.is_split);

	wire[`NT_M1:0] tmc_new_mask;
	wire all_threads = `NT < VX_gpu_inst_req.a_reg_data[0];
	genvar curr_t;
	generate
	for (curr_t = 0; curr_t < `NT; curr_t=curr_t+1) begin : tmc_new_mask_init
		assign tmc_new_mask[curr_t] = all_threads ? 1 : curr_t < VX_gpu_inst_req.a_reg_data[0];
	end
	endgenerate

	wire valid_inst = (|curr_valids);

	assign VX_warp_ctl.warp_num    = VX_gpu_inst_req.warp_num;
	assign VX_warp_ctl.change_mask = (VX_gpu_inst_req.is_tmc) && valid_inst;
	assign VX_warp_ctl.thread_mask = VX_gpu_inst_req.is_tmc ? tmc_new_mask : 0;

	// assign VX_warp_ctl.ebreak = (VX_gpu_inst_req.a_reg_data[0] == 0) && valid_inst;
	assign VX_warp_ctl.ebreak = VX_warp_ctl.change_mask && (VX_warp_ctl.thread_mask == 0);


	wire       wspawn     = VX_gpu_inst_req.is_wspawn;
	wire[31:0] wspawn_pc  = VX_gpu_inst_req.rd2;
	wire       all_active = `NW < VX_gpu_inst_req.a_reg_data[0];
	wire[`NW-1:0] wspawn_new_active;
	genvar curr_w;
	generate
	for (curr_w = 0; curr_w < `NW; curr_w=curr_w+1) begin : wspawn_new_active_init
		assign wspawn_new_active[curr_w] = all_active ? 1 : curr_w < VX_gpu_inst_req.a_reg_data[0];
	end
	endgenerate


	assign VX_warp_ctl.is_barrier = VX_gpu_inst_req.is_barrier && valid_inst;
	assign VX_warp_ctl.barrier_id = VX_gpu_inst_req.a_reg_data[0];

	wire[31:0] num_warps_m1       = VX_gpu_inst_req.rd2 - 1;
	assign VX_warp_ctl.num_warps  = num_warps_m1[$clog2(`NW):0];

	assign VX_warp_ctl.wspawn            = wspawn;
	assign VX_warp_ctl.wspawn_pc         = wspawn_pc;
	assign VX_warp_ctl.wspawn_new_active = wspawn_new_active;

	wire[`NT_M1:0] split_new_use_mask;
	wire[`NT_M1:0] split_new_later_mask;

	// VX_gpu_inst_req.pc
	genvar curr_s_t;
	generate
	for (curr_s_t = 0; curr_s_t < `NT; curr_s_t=curr_s_t+1) begin : masks_init
		wire curr_bool = (VX_gpu_inst_req.a_reg_data[curr_s_t] == 32'b1);

		assign split_new_use_mask[curr_s_t]   = curr_valids[curr_s_t] & (curr_bool);
		assign split_new_later_mask[curr_s_t] = curr_valids[curr_s_t] & (!curr_bool);
	end
	endgenerate

	wire[$clog2(`NT):0] num_valids;

	VX_countones #(.N(`NT)) valids_counter (
		.valids(curr_valids),
		.count (num_valids)
		);

	// wire[`NW_M1:0] num_valids = $countones(curr_valids);

	
	assign VX_warp_ctl.is_split         = is_split && (num_valids > 1);
	assign VX_warp_ctl.dont_split       = VX_warp_ctl.is_split && ((split_new_use_mask == 0) || (split_new_use_mask == {`NT{1'b1}}));
	assign VX_warp_ctl.split_new_mask   = split_new_use_mask;
	assign VX_warp_ctl.split_later_mask = split_new_later_mask;
	assign VX_warp_ctl.split_save_pc    = VX_gpu_inst_req.pc_next;
	assign VX_warp_ctl.split_warp_num   = VX_gpu_inst_req.warp_num;

	// VX_gpu_inst_req.is_wspawn
	// VX_gpu_inst_req.is_split
	// VX_gpu_inst_req.is_barrier

endmodule