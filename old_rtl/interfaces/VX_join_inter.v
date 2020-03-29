
`include "../VX_define.v"

`ifndef VX_JOIN_INTER

`define VX_JOIN_INTER

interface VX_join_inter ();

	wire            is_join;
	wire[`NW_M1:0]  join_warp_num;


endinterface


`endif