`ifndef VX_DCACHE_RSP
`define VX_DCACHE_RSP

`include "../VX_define.vh"

interface VX_dcache_response_inter ();

	wire [`NUM_THREADS-1:0][31:0] in_cache_driver_out_data;
	wire                          delay;

endinterface

`endif