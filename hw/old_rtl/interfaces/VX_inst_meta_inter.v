`include "../VX_define.v"

`ifndef VX_F_D_INTER

`define VX_F_D_INTER

interface VX_inst_meta_inter ();
	wire[31:0]       instruction;
	wire[31:0]       inst_pc;
	wire[`NW_M1:0]   warp_num;
	wire[`NT_M1:0]   valid;

endinterface


`endif