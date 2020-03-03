`include "VX_cache_config.v"

module VX_bank (
	input wire clk,
	input wire reset,

	// Input Core Request
	input wire [`NUMBER_REQUESTS-1:0]             bank_valids,
	input wire [`NUMBER_REQUESTS-1:0][31:0]       bank_addr,
	input wire [`NUMBER_REQUESTS-1:0][31:0]       bank_writedata,
	input wire [4:0]                              bank_rd,
	input wire [1:0]                              bank_wb,
	input wire [`NW_M1:0]                         bank_warp_num,
	input wire [2:0]                              bank_mem_read,  
	input wire [2:0]                              bank_mem_write,

	// Output Core WB
	input  wire                                   bank_wb_pop,
	output wire [`vx_clog2(`NUMBER_REQUESTS)-1:0] bank_wb_tid,
	output wire [4:0]                             bank_wb_rd,
	output wire [1:0]                             bank_wb_wb,
	output wire [`NW_M1:0]                        bank_wb_warp_num,
	output wire [31:0]                            bank_wb_data,

	// Dram Fill Requests
	output wire                                   dram_fill_req,
	output wire[31:0]                             dram_fill_req_addr,
	input  wire                                   dram_fill_req_queue_full,

	// Dram Fill Response
	input  wire                                   dram_fill_rsp,
	input  wire [31:0]                            dram_fill_addr,
	input  wire[`BANK_LINE_SIZE_RNG][31:0]        dram_fill_rsp_data,

	// Dram WB Requests
	input  wire                                   dram_wb_queue_pop,
	output wire                                   dram_wb_req,
	output wire[31:0]                             dram_wb_req_addr,
	output wire[`BANK_LINE_SIZE_RNG][31:0]        dram_wb_req_data
);

endmodule