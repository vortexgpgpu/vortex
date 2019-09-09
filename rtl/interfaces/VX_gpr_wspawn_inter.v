
`ifndef VX_GPR_WSPAWN_INTER

`define VX_GPR_WSPAWN_INTER


interface VX_gpr_wspawn_inter ();

	wire           is_wspawn;
	wire[`NW_M1:0] which_wspawn;
	// wire[`NW_M1:0] warp_num;


	modport snk (
		input is_wspawn,
		input which_wspawn
	);


	modport src (
		output is_wspawn,
		output which_wspawn
	);

endinterface



`endif