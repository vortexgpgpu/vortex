
`include "VX_define.v"

`ifndef VX_DCACHE_RSP

`define VX_DCACHE_RSP

interface VX_dcache_response_inter ();

		wire[31:0]  in_cache_driver_out_data[`NT_M1:0];

	// source-side view
	modport snk (
		input in_cache_driver_out_data
	);


	// source-side view
	modport src (
		output in_cache_driver_out_data
	);


endinterface


`endif