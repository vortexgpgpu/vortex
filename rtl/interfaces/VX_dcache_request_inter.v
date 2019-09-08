
`include "VX_define.v"

`ifndef VX_DCACHE_REQ

`define VX_DCACHE_REQ

interface VX_dcache_request_inter ();

		wire[31:0] out_cache_driver_in_address[`NT_M1:0];
		wire[2:0]  out_cache_driver_in_mem_read;
		wire[2:0]  out_cache_driver_in_mem_write;
		wire       out_cache_driver_in_valid[`NT_M1:0];
		wire[31:0] out_cache_driver_in_data[`NT_M1:0];

	// source-side view
	modport snk (
		input out_cache_driver_in_address,
		input out_cache_driver_in_mem_read,
		input out_cache_driver_in_mem_write,
		input out_cache_driver_in_valid,
		input out_cache_driver_in_data
	);


	// source-side view
	modport src (
		output out_cache_driver_in_address,
		output out_cache_driver_in_mem_read,
		output out_cache_driver_in_mem_write,
		output out_cache_driver_in_valid,
		output out_cache_driver_in_data
	);


endinterface


`endif