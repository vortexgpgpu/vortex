
`include "VX_define.v"

`ifndef VX_FWD_REQ

`define VX_FWD_REQ

interface VX_forward_reqeust_inter ();

		wire[4:0]      src1;
		wire[4:0]      src2;
		wire[`NW_M1:0] warp_num;

	// source-side view
	modport snk (
		input src1,
		input src2,
		input warp_num
	);


	// source-side view
	modport src (
		output src1,
		output src2,
		output warp_num
	);


endinterface


`endif