`include "VX_define.v"

`ifndef VX_F_D_INTER

`define VX_F_D_INTER

interface VX_inst_meta_inter ();
	wire[31:0]       instruction;
	wire[31:0]       inst_pc;
	wire[`NW_M1:0]   warp_num;
	wire[`NT_M1:0]   valid;

// source-side view
modport snk (
	input instruction,
	input inst_pc,
	input warp_num,
	input valid
);

// sink-side view
modport src (
	output instruction,
	output inst_pc,
	output warp_num,
	output valid
);

endinterface


`endif