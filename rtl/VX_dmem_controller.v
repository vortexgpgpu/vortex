
`include "VX_define.v"

module VX_dmem_controller (
	input wire               clk,

	// MEM-Processor
	VX_dcache_request_inter  VX_dcache_req,
	VX_dcache_response_inter VX_dcache_rsp
);


	wire to_shm = VX_dcache_req.out_cache_driver_in_address[0][31:24] == 8'hFF;

	wire[`NT_M1:0]       sm_driver_in_valid     = VX_dcache_req.out_cache_driver_in_valid & {`NT{to_shm}};
	wire[`NT_M1:0]       cache_driver_in_valid  = VX_dcache_req.out_cache_driver_in_valid & {`NT{~to_shm}};




	wire[`NT_M1:0][31:0] cache_driver_in_address   = VX_dcache_req.out_cache_driver_in_address;
	wire[2:0]            cache_driver_in_mem_read  = VX_dcache_req.out_cache_driver_in_mem_read;
	wire[2:0]            cache_driver_in_mem_write = VX_dcache_req.out_cache_driver_in_mem_write;
	wire[`NT_M1:0][31:0] cache_driver_in_data      = VX_dcache_req.out_cache_driver_in_data;


	wire[`NT_M1:0][31:0] cache_driver_out_data;
	wire[`NT_M1:0]       cache_driver_out_valid; // Not used for now
	wire                 delay;


	VX_shared_memory #(.NB(7), .BITS_PER_BANK(3)) shared_memory (
		.clk       (clk),
		.in_valid  (sm_driver_in_valid),
		.in_address(cache_driver_in_address),
		.in_data   (cache_driver_in_data),
		.mem_read  (cache_driver_in_mem_read),
		.mem_write (cache_driver_in_mem_write),
		.out_valid (cache_driver_out_valid),
		.out_data  (cache_driver_out_data),
		.stall     (delay)
		);




	assign VX_dcache_rsp.in_cache_driver_out_data = cache_driver_out_data;
	assign VX_dcache_rsp.delay                    = delay;


endmodule