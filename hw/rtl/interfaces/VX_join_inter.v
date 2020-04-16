
`include "../VX_define.vh"

`ifndef VX_JOIN_INTER

`define VX_JOIN_INTER

interface VX_join_inter ();

	wire            is_join;
	wire[`NW_BITS-1:0]  join_warp_num;


endinterface


`endif