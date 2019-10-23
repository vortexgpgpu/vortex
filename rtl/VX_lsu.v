
`include "VX_define.v"


module VX_lsu (
		// input wire               clk,
		VX_lsu_req_inter         VX_lsu_req,

		// Write back to GPR
		VX_inst_mem_wb_inter     VX_mem_wb,

		VX_dcache_response_inter VX_dcache_rsp,
		VX_dcache_request_inter  VX_dcache_req,
		output wire              out_delay
	);

	// VX_inst_mem_wb_inter VX_mem_wb_temp();

	assign out_delay = VX_dcache_rsp.delay;


	// Generate Addresses
	wire[`NT_M1:0][31:0] address;
	VX_lsu_addr_gen VX_lsu_addr_gen
	(
		.base_address(VX_lsu_req.base_address),
		.offset      (VX_lsu_req.offset),
		.address     (address)
	);


	genvar index;
	for (index = 0; index <= `NT_M1; index = index + 1) begin
		assign VX_dcache_req.out_cache_driver_in_address[index]   = address[index];
		assign VX_dcache_req.out_cache_driver_in_data[index]      = VX_lsu_req.store_data[index];
		assign VX_dcache_req.out_cache_driver_in_valid[index]     = (VX_lsu_req.valid[index] && !VX_dcache_rsp.delay);

		assign VX_mem_wb.loaded_data[index]                       = VX_dcache_rsp.in_cache_driver_out_data[index];
	end

		assign VX_dcache_req.out_cache_driver_in_mem_read  = VX_lsu_req.mem_read;
		assign VX_dcache_req.out_cache_driver_in_mem_write = VX_lsu_req.mem_write;


		assign VX_mem_wb.rd          = VX_lsu_req.rd;
		assign VX_mem_wb.wb          = VX_lsu_req.wb;
		assign VX_mem_wb.wb_valid    = VX_lsu_req.valid;
		assign VX_mem_wb.wb_warp_num = VX_lsu_req.warp_num;



	// wire zero_temp = 0;
	// VX_generic_register #(.N(256)) register_wb_data 
	// (
	// 	.clk  (clk),
	// 	.reset(zero_temp),
	// 	.stall(zero_temp),
	// 	.flush(zero_temp),
	// 	.in   ({VX_mem_wb_temp.loaded_data, VX_mem_wb_temp.rd, VX_mem_wb_temp.wb, VX_mem_wb_temp.wb_valid, VX_mem_wb_temp.wb_warp_num}),
	// 	.out  ({VX_mem_wb.loaded_data     , VX_mem_wb.rd     , VX_mem_wb.wb     , VX_mem_wb.wb_valid     , VX_mem_wb.wb_warp_num     })
	// );


endmodule // Memory


