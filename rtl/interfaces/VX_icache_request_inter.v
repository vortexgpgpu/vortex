
`include "VX_define.v"

`ifndef VX_ICACHE_REQ

`define VX_ICACHE_REQ

interface VX_icache_request_inter ();

	wire[31:0] pc_address;

	// source-side view
	modport snk (
		input pc_address
	);


	// source-side view
	modport src (
		output pc_address
	);


endinterface


`endif