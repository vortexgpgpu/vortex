
`include "../VX_define.v"

`ifndef VX_DCACHE_RSP

`define VX_DCACHE_RSP

interface VX_dcache_response_inter ();

		wire[31:0]  in_cache_driver_out_data[`NT_M1:0];

endinterface


`endif