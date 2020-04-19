
`ifndef VX_ICACHE_REQ
`define VX_ICACHE_REQ

`include "../VX_define.vh"

interface VX_icache_request_inter ();

	wire [31:0]		pc_address;
	wire [2:0]      out_cache_driver_in_mem_read;
	wire [2:0]      out_cache_driver_in_mem_write;
	wire       		out_cache_driver_in_valid;
	wire [31:0] 	out_cache_driver_in_data;

endinterface

`endif