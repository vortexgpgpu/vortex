
`include "../VX_define.v"

`ifndef VX_BRANCH_RSP

`define VX_BRANCH_RSP

interface VX_branch_response_inter ();
	wire           valid_branch;
	wire           branch_dir;
	wire[31:0]     branch_dest;
	wire[`NW_M1:0] branch_warp_num;


endinterface


`endif