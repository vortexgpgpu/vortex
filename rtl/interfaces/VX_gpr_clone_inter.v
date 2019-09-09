
`ifndef VX_GPR_CLONE_INTER

`define VX_GPR_CLONE_INTER


interface VX_gpr_clone_inter ();

	wire           is_clone;
	wire[`NW_M1:0] warp_num;


	modport snk (
		input is_clone,
		input warp_num
	);


	modport src (
		output is_clone,
		output warp_num
	);

endinterface



`endif