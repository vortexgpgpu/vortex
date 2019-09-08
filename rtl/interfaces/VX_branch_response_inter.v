
`include "VX_define.v"

`ifndef VX_BRANCH_RSP

`define VX_BRANCH_RSP

interface VX_branch_response_inter ();

	wire           branch_dir;
	wire[31:0]     branch_dest;
	wire[`NW_M1:0] branch_warp_num;

	// source-side view
	modport snk (
		input branch_dir,
		input branch_dest
	);


	// source-side view
	modport src (
		output branch_dir,
		output branch_dest
	);


endinterface


`endif