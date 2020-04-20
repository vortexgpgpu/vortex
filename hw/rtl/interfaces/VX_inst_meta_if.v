`ifndef VX_F_D_INTER
`define VX_F_D_INTER

`include "../VX_define.vh"

interface VX_inst_meta_if ();

	wire [31:0]       		instruction;
	wire [31:0]       		inst_pc;
	wire [`NW_BITS-1:0]   	warp_num;
	wire [`NUM_THREADS-1:0]	valid;

endinterface

`endif