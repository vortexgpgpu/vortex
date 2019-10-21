module VX_gpgpu_inst (
	// Input
	VX_gpu_inst_req_inter    VX_gpu_inst_req,

	// Output
	VX_warp_ctl_inter        VX_warp_ctl
);


	wire[`NT_M1:0] tmc_new_mask;
	genvar curr_t;
	for (curr_t = 0; curr_t < `NT; curr_t=curr_t+1)
	begin
		assign tmc_new_mask[curr_t] = curr_t < VX_gpu_inst_req.a_reg_data[0];
	end

	wire valid_inst = (|VX_gpu_inst_req.valid);

	assign VX_warp_ctl.warp_num    = VX_gpu_inst_req.warp_num;
	assign VX_warp_ctl.change_mask = (VX_gpu_inst_req.is_tmc || VX_gpu_inst_req.is_split) && valid_inst;
	assign VX_warp_ctl.thread_mask = VX_gpu_inst_req.is_tmc ? tmc_new_mask : 0;

	assign VX_warp_ctl.ebreak = (VX_gpu_inst_req.a_reg_data[0] == 0) && valid_inst;

	assign VX_warp_ctl.wspawn = 0;
	assign VX_warp_ctl.wspawn_pc = 0;


	wire[`NT_M1:0] split_new_use_mask;
	wire[`NT_M1:0] split_new_later_mask;

	// VX_gpu_inst_req.pc
	genvar curr_s_t;
	for (curr_s_t = 0; curr_s_t < `NT; curr_s_t=curr_s_t+1) begin
		wire curr_bool = (VX_gpu_inst_req.a_reg_data[curr_s_t] == 32'b1);

		assign split_new_use_mask[curr_s_t]   = VX_gpu_inst_req.valid[curr_s_t] & (curr_bool);
		assign split_new_later_mask[curr_s_t] = VX_gpu_inst_req.valid[curr_s_t] & (!curr_bool);
	end

	reg[$clog2(`NT)-1:0] num_valids;
	integer z;
	always @(*) begin
		num_valids = 0;
		for (z = 0; z < `NT; z=z+1) begin
			if (VX_gpu_inst_req.valid[z]) num_valids = num_valids + 1;
		end
	end
	
	assign VX_warp_ctl.is_split         = (VX_gpu_inst_req.is_split) && (num_valids > 1);
	assign VX_warp_ctl.split_new_mask   = split_new_use_mask;
	assign VX_warp_ctl.split_later_mask = split_new_later_mask;
	assign VX_warp_ctl.split_save_pc    = VX_gpu_inst_req.pc_next;
	assign VX_warp_ctl.split_warp_num   = VX_gpu_inst_req.warp_num;

	// VX_gpu_inst_req.is_wspawn
	// VX_gpu_inst_req.is_split
	// VX_gpu_inst_req.is_barrier

endmodule