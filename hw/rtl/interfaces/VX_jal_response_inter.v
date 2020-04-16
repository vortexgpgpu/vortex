
`include "../VX_define.vh"

`ifndef VX_JAL_RSP

`define VX_JAL_RSP

interface VX_jal_response_inter ();

	wire           jal;
	wire[31:0]     jal_dest;
	wire[`NW_BITS-1:0] jal_warp_num;
	
endinterface


`endif