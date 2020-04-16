
`ifndef VX_JOIN_INTER
`define VX_JOIN_INTER

`include "../VX_define.vh"

interface VX_join_inter ();

	wire            is_join;
	wire[`NW_BITS-1:0]  join_warp_num;

endinterface

`endif