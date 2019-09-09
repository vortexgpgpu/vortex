
`ifndef VX_GPR_READ

`define VX_GPR_READ


interface VX_gpr_read_inter ();

	wire[4:0]      rs1;
	wire[4:0]      rs2;
	wire[`NW_M1:0] warp_num;

	modport snk (
		input rs1,
		input rs2,
		input warp_num
	);


	modport src (
		output rs1,
		output rs2,
		output warp_num
	);

endinterface



`endif