`ifndef VX_GPR_READ
`define VX_GPR_READ

`include "../VX_define.vh"

interface VX_gpr_read_inter ();

	wire [4:0]      	rs1;
	wire [4:0]      	rs2;
	wire [`NW_BITS-1:0] warp_num;

endinterface

`endif