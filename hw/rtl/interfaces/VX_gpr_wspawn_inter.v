`include "../VX_define.vh"
`ifndef VX_GPR_WSPAWN_INTER

`define VX_GPR_WSPAWN_INTER


interface VX_gpr_wspawn_inter ();
	/* verilator lint_off UNUSED */
	wire           is_wspawn;
	wire[`NW_BITS-1:0] which_wspawn;
	// wire[`NW_BITS-1:0] warp_num;
	/* verilator lint_on UNUSED */

endinterface



`endif