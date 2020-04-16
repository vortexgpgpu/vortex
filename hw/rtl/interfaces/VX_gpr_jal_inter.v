`ifndef VX_GPR_JAL_INTER
`define VX_GPR_JAL_INTER

`include "../VX_define.vh"

interface VX_gpr_jal_inter ();

	wire       is_jal;
	wire[31:0] curr_PC;
	
endinterface

`endif