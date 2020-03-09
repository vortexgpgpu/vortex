`include "VX_define.v"

module VX_lsu (
		input wire               clk,
		input wire               reset,
		input wire               no_slot_mem,
		VX_lsu_req_inter         VX_lsu_req,

		// Write back to GPR
		VX_inst_mem_wb_inter     VX_mem_wb,

		VX_gpu_dcache_res_inter  VX_dcache_rsp,
		VX_gpu_dcache_req_inter  VX_dcache_req,
		output wire              out_delay
	);

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


	// Core Request
	assign VX_dcache_req.core_req_valid      = use_valid;
	assign VX_dcache_req.core_req_addr       = use_address;
	assign VX_dcache_req.core_req_writedata  = use_store_data;
	assign VX_dcache_req.core_req_mem_read   = {`NT{use_mem_read}};
	assign VX_dcache_req.core_req_mem_write  = {`NT{use_mem_write}};
	assign VX_dcache_req.core_req_rd         = use_rd;
	assign VX_dcache_req.core_req_wb         = {`NT{use_wb}};
	assign VX_dcache_req.core_req_warp_num   = use_warp_num;
	assign VX_dcache_req.core_req_pc         = use_pc;

	// Core can't accept response
	assign VX_dcache_req.core_no_wb_slot     = no_slot_mem;
	

	// Cache can't accept request
	assign out_delay = VX_dcache_rsp.delay_req;

	// Core Response
	assign VX_mem_wb.rd          = VX_dcache_rsp.core_wb_req_rd;
	assign VX_mem_wb.wb          = VX_dcache_rsp.core_wb_req_wb;
	assign VX_mem_wb.wb_valid    = VX_dcache_rsp.core_wb_valid;
	assign VX_mem_wb.wb_warp_num = VX_dcache_rsp.core_wb_warp_num;
	assign VX_mem_wb.loaded_data = VX_dcache_rsp.core_wb_readdata;
	assign VX_mem_wb.mem_wb_pc   = VX_dcache_rsp.core_wb_pc[0];





endmodule // Memory


