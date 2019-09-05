
`include "VX_define.v"

`ifndef VX_WARP_CTL_INTER

`define VX_WARP_CTL_INTER

interface VX_warp_ctl_inter ();

	wire[`NW_M1:0] warp_num;
	wire           change_mask;
	wire[`NT_M1:0] thread_mask;
	wire           wspawn;
	wire[31:0]     wspawn_pc;
	wire           ebreak;

	// source-side view
	modport snk (
		input warp_num,
		input change_mask,
		input thread_mask,
		input wspawn,
		input wspawn_pc,
		input ebreak
	);


	// source-side view
	modport src (
		output warp_num,
		output change_mask,
		output thread_mask,
		output wspawn,
		output wspawn_pc,
		output ebreak
	);


endinterface


`endif