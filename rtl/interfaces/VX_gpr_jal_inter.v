
`ifndef VX_GPR_JAL_INTER

`define VX_GPR_JAL_INTER


interface VX_gpr_jal_inter ();
	wire       is_jal;
	wire[31:0] curr_PC;

	modport snk (
		input is_jal,
		input curr_PC
	);


	modport src (
		output is_jal,
		output curr_PC
	);

endinterface



`endif