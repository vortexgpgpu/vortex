`ifndef VX_WSTALL_INTER
`define VX_WSTALL_INTER

`include "VX_define.vh"

interface VX_wstall_if();

	wire           		wstall;
	wire [`NW_BITS-1:0] warp_num;

endinterface

`endif