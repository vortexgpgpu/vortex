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


	assign VX_warp_ctl.warp_num    = VX_gpu_inst_req.warp_num;
	assign VX_warp_ctl.change_mask = (VX_gpu_inst_req.is_tmc || VX_gpu_inst_req.is_split) && (|VX_gpu_inst_req.valid);
	assign VX_warp_ctl.thread_mask = VX_gpu_inst_req.is_tmc ? tmc_new_mask : 0;

	assign VX_warp_ctl.ebreak = (VX_gpu_inst_req.a_reg_data[0] == 0);

	assign VX_warp_ctl.wspawn = 0;
	assign VX_warp_ctl.wspawn_pc = 0;


	// VX_gpu_inst_req.is_wspawn
	// VX_gpu_inst_req.is_split
	// VX_gpu_inst_req.is_barrier

endmodule