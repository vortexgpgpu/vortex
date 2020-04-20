`ifndef VX_DCACHE_RSP
`define VX_DCACHE_RSP

`include "../VX_define.v"

interface VX_dcache_response_if ();

	wire [`NUM_THREADS-1:0][31:0] in_cache_driver_out_data;
	wire                          delay;

endinterface

`endif