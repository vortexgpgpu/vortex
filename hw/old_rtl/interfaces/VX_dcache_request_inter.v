
`include "../VX_define.v"

`ifndef VX_DCACHE_REQ

`define VX_DCACHE_REQ

interface VX_dcache_request_inter ();

		wire[`NT_M1:0][31:0] out_cache_driver_in_address;
		wire[2:0]            out_cache_driver_in_mem_read;
		wire[2:0]            out_cache_driver_in_mem_write;
		wire[`NT_M1:0]       out_cache_driver_in_valid;
		wire[`NT_M1:0][31:0] out_cache_driver_in_data;

endinterface


`endif