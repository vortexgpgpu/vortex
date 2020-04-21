`ifndef VX_ICACHE_RSP
`define VX_ICACHE_RSP

`include "../VX_define.vh"

interface VX_icache_rsp_if ();

	// wire ready;
	// wire stall;
	wire [31:0]	instruction;
	wire       	delay;

endinterface

`endif