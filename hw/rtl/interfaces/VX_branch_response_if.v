`ifndef VX_BRANCH_RSP
`define VX_BRANCH_RSP

`include "../VX_define.v"

interface VX_branch_response_if ();

	wire           		valid_branch;
	wire           		branch_dir;
	wire [31:0]     	branch_dest;
	wire [`NW_BITS-1:0] branch_warp_num;

endinterface

`endif