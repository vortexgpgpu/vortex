`ifndef VX_ICACHE_RSP
`define VX_ICACHE_RSP

`include "../VX_define.v"

interface VX_icache_response_if ();

	// wire ready;
	// wire stall;
	wire [31:0]	instruction;
	wire       	delay;

endinterface

`endif