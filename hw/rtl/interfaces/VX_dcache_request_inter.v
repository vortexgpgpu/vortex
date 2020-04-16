`ifndef VX_DCACHE_REQ
`define VX_DCACHE_REQ

`include "../VX_define.vh"

interface VX_dcache_request_inter ();

		wire[`NUM_THREADS-1:0][31:0] out_cache_driver_in_address;
		wire[2:0]            out_cache_driver_in_mem_read;
		wire[2:0]            out_cache_driver_in_mem_write;
		wire[`NUM_THREADS-1:0]       out_cache_driver_in_valid;
		wire[`NUM_THREADS-1:0][31:0] out_cache_driver_in_data;

endinterface

`endif