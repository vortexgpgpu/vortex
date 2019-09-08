
`include "VX_define.v"

`ifndef VX_JAL_RSP

`define VX_JAL_RSP

interface VX_jal_response_inter ();

	wire           jal;
	wire[31:0]     jal_dest;
	wire[`NW_M1:0] jal_warp_num;

	// source-side view
	modport snk (
		input jal,
		input jal_dest,
		input jal_warp_num
	);


	// source-side view
	modport src (
		output jal,
		output jal_dest,
		output jal_warp_num
	);


endinterface


`endif