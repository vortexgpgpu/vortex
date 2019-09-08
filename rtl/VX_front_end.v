
module VX_front_end (
	input wire clk,
	input wire reset,
	
);



VX_fetch vx_fetch(
		.clk                (clk),
		.reset              (reset),
		.in_branch_dir      (memory_branch_dir),
		.in_freeze          (total_freeze),
		.in_branch_dest     (memory_branch_dest),
		.in_branch_stall    (decode_branch_stall),
		.in_fwd_stall       (forwarding_fwd_stall),
		.in_branch_stall_exe(execute_branch_stall),
		.in_clone_stall     (decode_clone_stall),
		.in_jal             (e_m_jal),
		.in_jal_dest        (e_m_jal_dest),
		.in_interrupt       (interrupt),
		.in_debug           (debug),
		.in_memory_warp_num (VX_mem_wb.warp_num),
		.icache_response    (icache_response_fe),
		.VX_warp_ctl        (VX_warp_ctl),

		.icache_request     (icache_request_fe),
		.out_delay          (fetch_delay),
		.out_ebreak         (fetch_ebreak),
		.out_which_wspawn   (fetch_which_warp),
		.fe_inst_meta_fd    (fe_inst_meta_fd)
	);


VX_f_d_reg vx_f_d_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_freeze      (total_freeze),
		.in_clone_stall (decode_clone_stall),
		.fe_inst_meta_fd(fe_inst_meta_fd),
		.fd_inst_meta_de(fd_inst_meta_de)
	);


VX_decode vx_decode(
		.clk               (clk),
		.fd_inst_meta_de   (fd_inst_meta_de),
		.VX_writeback_inter(VX_writeback_inter),
		.VX_fwd_rsp        (VX_fwd_rsp),
		.in_which_wspawn   (fetch_which_warp),

		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_fwd_req_de     (VX_fwd_req_de),
		.VX_warp_ctl       (VX_warp_ctl),
		.out_clone_stall   (decode_clone_stall),
		.out_branch_stall  (decode_branch_stall)
	);


VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_freeze      (total_freeze),
		.in_clone_stall (decode_clone_stall),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);


endmodule