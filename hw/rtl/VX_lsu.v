`include "VX_define.vh"

module VX_lsu (
	input wire               clk,
	input wire               reset,
	input wire               no_slot_mem,
	VX_lsu_req_inter         vx_lsu_req,

	// Write back to GPR
	VX_inst_mem_wb_inter     vx_mem_wb,

	VX_gpu_dcache_rsp_inter  vx_dcache_rsp,
	VX_gpu_dcache_req_inter  vx_dcache_req,
	output wire              out_delay
);
	// Generate Addresses
	wire[`NUM_THREADS-1:0][31:0] address;
	VX_lsu_addr_gen VX_lsu_addr_gen	(
		.base_address (vx_lsu_req.base_address),
		.offset       (vx_lsu_req.offset),
		.address      (address)
	);

	wire[`NUM_THREADS-1:0][31:0] 	use_address;
	wire[`NUM_THREADS-1:0][31:0] 	use_store_data;
	wire[`NUM_THREADS-1:0]       	use_valid;
	wire[2:0]            use_mem_read; 
	wire[2:0]            use_mem_write;
	wire[4:0]            use_rd;
	wire[`NW_BITS-1:0]   use_warp_num;
	wire[1:0]            use_wb;
	wire[31:0]           use_pc;	

	wire zero = 0;

	VX_generic_register #(
		.N(45 + `NW_BITS-1 + 1 + `NUM_THREADS*65)
	) lsu_buffer(
		.clk  (clk),
		.reset(reset),
		.stall(out_delay),
		.flush(zero),
		.in   ({address    , vx_lsu_req.store_data, vx_lsu_req.valid, vx_lsu_req.mem_read, vx_lsu_req.mem_write, vx_lsu_req.rd, vx_lsu_req.warp_num, vx_lsu_req.wb, vx_lsu_req.lsu_pc}),
		.out  ({use_address, use_store_data       , use_valid       , use_mem_read       , use_mem_write       , use_rd       , use_warp_num       , use_wb       , use_pc           })
	);

	// Core Request
	assign vx_dcache_req.core_req_valid      = use_valid;
	assign vx_dcache_req.core_req_addr       = use_address;
	assign vx_dcache_req.core_req_writedata  = use_store_data;
	assign vx_dcache_req.core_req_mem_read   = {`NUM_THREADS{use_mem_read}};
	assign vx_dcache_req.core_req_mem_write  = {`NUM_THREADS{use_mem_write}};
	assign vx_dcache_req.core_req_rd         = use_rd;
	assign vx_dcache_req.core_req_wb         = {`NUM_THREADS{use_wb}};
	assign vx_dcache_req.core_req_warp_num   = use_warp_num;
	assign vx_dcache_req.core_req_pc         = use_pc;

	// Core can't accept response
	assign vx_dcache_req.core_no_wb_slot     = no_slot_mem;	

	// Cache can't accept request
	assign out_delay = vx_dcache_rsp.delay_req;

	// Core Response
	assign vx_mem_wb.rd          = vx_dcache_rsp.core_wb_req_rd;
	assign vx_mem_wb.wb          = vx_dcache_rsp.core_wb_req_wb;
	assign vx_mem_wb.wb_valid    = vx_dcache_rsp.core_wb_valid;
	assign vx_mem_wb.wb_warp_num = vx_dcache_rsp.core_wb_warp_num;
	assign vx_mem_wb.loaded_data = vx_dcache_rsp.core_wb_readdata;
	
	wire[(`LOG2UP(`NUM_THREADS))-1:0] use_pc_index;

`DEBUG_BEGIN
	wire found;
`DEBUG_END

	VX_generic_priority_encoder #(.N(`NUM_THREADS)) pick_first_pc(
		.valids(vx_dcache_rsp.core_wb_valid),
		.index (use_pc_index),
		.found (found)
		);

	assign vx_mem_wb.mem_wb_pc   = vx_dcache_rsp.core_wb_pc[use_pc_index];
	
endmodule // Memory



