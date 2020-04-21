
`ifndef VX_JAL_RSP
`define VX_JAL_RSP

`include "../VX_define.vh"

interface VX_jal_rsp_if ();

	wire           		jal;
	wire [31:0]     	jal_dest;
	wire [`NW_BITS-1:0] jal_warp_num;
	
endinterface

`endif