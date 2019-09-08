`include "VX_define.v"

module VX_front_end (
	input wire clk,
	input wire reset,

	input wire total_freeze,
	input wire forwarding_fwd_stall,

	input wire[`NW_M1:0] fetch_which_warp,
	input wire           execute_branch_stall,


	VX_warp_ctl_inter         VX_warp_ctl,
	VX_inst_meta_inter        fe_inst_meta_fd,
	VX_wb_inter               VX_writeback_inter,
	VX_forward_reqeust_inter  VX_fwd_req_de,
	VX_forward_response_inter VX_fwd_rsp,
	VX_frE_to_bckE_req_inter  VX_bckE_req,


	output wire decode_clone_stall,
	output wire decode_branch_stall,
	output wire[11:0] decode_csr_address
);


VX_frE_to_bckE_req_inter VX_frE_to_bckE_req();
VX_inst_meta_inter       fd_inst_meta_de();

// From decode
wire           internal_decode_branch_stall;
wire           internal_decode_clone_stall;

assign decode_clone_stall  = internal_decode_clone_stall;
assign decode_branch_stall = internal_decode_branch_stall; 

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
		.out_clone_stall   (internal_decode_clone_stall),
		.out_branch_stall  (internal_decode_branch_stall)
	);


VX_d_e_reg vx_d_e_reg(
		.clk            (clk),
		.reset          (reset),
		.in_fwd_stall   (forwarding_fwd_stall),
		.in_branch_stall(execute_branch_stall),
		.in_freeze      (total_freeze),
		.in_clone_stall (internal_decode_clone_stall),
		.VX_frE_to_bckE_req(VX_frE_to_bckE_req),
		.VX_bckE_req       (VX_bckE_req)
	);


assign decode_csr_address = VX_frE_to_bckE_req.csr_address;


endmodule


