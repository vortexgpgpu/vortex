
`include "../VX_define.v"

`ifndef VX_gpr_data_INTER

`define VX_gpr_data_INTER

interface VX_gpr_data_inter ();
	wire[`NT_M1:0][31:0] a_reg_data;
	wire[`NT_M1:0][31:0] b_reg_data;
endinterface


`endif