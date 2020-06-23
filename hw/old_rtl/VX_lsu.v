
`include "VX_define.v"


module VX_lsu (
		input wire               clk,
		input wire               reset,
		input wire               no_slot_mem,
		VX_lsu_req_inter         VX_lsu_req,

		// Write back to GPR
		VX_inst_mem_wb_inter     VX_mem_wb,

		VX_dcache_response_inter VX_dcache_rsp,
		VX_dcache_request_inter  VX_dcache_req,
		output wire              out_delay
	);

	// VX_inst_mem_wb_inter VX_mem_wb_temp();

	assign out_delay = VX_dcache_rsp.delay || no_slot_mem;


	// Generate Addresses
	wire[`NT_M1:0][31:0] address;
	VX_lsu_addr_gen VX_lsu_addr_gen
	(
		.base_address(VX_lsu_req.base_address),
		.offset      (VX_lsu_req.offset),
		.address     (address)
	);


	wire[`NT_M1:0][31:0] use_address;
	wire[`NT_M1:0][31:0] use_store_data;
	wire[`NT_M1:0]       use_valid;
	wire[2:0]            use_mem_read; 
	wire[2:0]            use_mem_write;
	wire[4:0]            use_rd;
	wire[`NW_M1:0]       use_warp_num;
	wire[1:0]            use_wb;
	wire[31:0]           use_pc;

	

	wire zero = 0;

	VX_generic_register #(.N(45 + `NW_M1 + 1 + `NT*65)) lsu_buffer(
		.clk  (clk),
		.reset(reset),
		.stall(out_delay),
		.flush(zero),
		.in   ({address    , VX_lsu_req.store_data, VX_lsu_req.valid, VX_lsu_req.mem_read, VX_lsu_req.mem_write, VX_lsu_req.rd, VX_lsu_req.warp_num, VX_lsu_req.wb, VX_lsu_req.lsu_pc}),
		.out  ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
		);


	genvar index;
	for (index = 0; index <= `NT_M1; index = index + 1) begin
		assign VX_dcache_req.out_cache_driver_in_address[index]   = use_address[index];
		assign VX_dcache_req.out_cache_driver_in_data[index]      = use_store_data[index];
		assign VX_dcache_req.out_cache_driver_in_valid[index]     = (use_valid[index]);

		assign VX_mem_wb.loaded_data[index]                       = VX_dcache_rsp.in_cache_driver_out_data[index];
	end

		assign VX_dcache_req.out_cache_driver_in_mem_read  = use_mem_read;
		assign VX_dcache_req.out_cache_driver_in_mem_write = use_mem_write;


		assign VX_mem_wb.rd          = use_rd;
		assign VX_mem_wb.wb          = use_wb & {!VX_dcache_rsp.delay, !VX_dcache_rsp.delay};
		assign VX_mem_wb.wb_valid    = use_valid;
		assign VX_mem_wb.wb_warp_num = use_warp_num;

		assign VX_mem_wb.mem_wb_pc   = use_pc;

		// integer curr_t;
		// always @(negedge clk) begin
		// 	for (int curr_t = 0; curr_t < `NT; curr_t=curr_t+1)
		// 	if ((VX_dcache_req.out_cache_driver_in_valid[curr_t]) && !out_delay) begin
		// 		if (VX_dcache_req.out_cache_driver_in_mem_read != `NO_MEM_READ) begin
		// 			$display("Reading addr: %x val: %x", address[0], VX_mem_wb.loaded_data[0]);
		// 		end

		// 		if (VX_dcache_req.out_cache_driver_in_mem_write != `NO_MEM_WRITE) begin
		// 			$display("Writing addr: %x val: %x", address[0], VX_dcache_req.out_cache_driver_in_data[0]);
		// 		end
		// 	end
		// end

	// wire zero_temp = 0;
	// VX_generic_register #(.N(142)) register_wb_data 
	// (
	// 	.clk  (clk),
	// 	.reset(reset),
	// 	.stall(zero_temp),
	// 	.flush(out_delay),
	// 	.in   ({VX_mem_wb_temp.loaded_data, VX_mem_wb_temp.rd, VX_mem_wb_temp.wb, VX_mem_wb_temp.wb_valid, VX_mem_wb_temp.wb_warp_num}),
	// 	.out  ({VX_mem_wb.loaded_data     , VX_mem_wb.rd     , VX_mem_wb.wb     , VX_mem_wb.wb_valid     , VX_mem_wb.wb_warp_num     })
	// );


endmodule // Memory


