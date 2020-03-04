`include "VX_cache_config.v"


module VX_cache_dram_req_arb (
	input  wire clk,
	input  wire reset,


	// Fill Request
    output wire                                             dfqq_full,
    input  wire[`NUMBER_BANKS-1:0]                          per_bank_dram_fill_req,
    input  wire[`NUMBER_BANKS-1:0][31:0]                    per_bank_dram_fill_req_addr,

    // DFQ Request
    output wire[`NUMBER_BANKS-1:0]                            per_bank_dram_wb_queue_pop,
    input  wire[`NUMBER_BANKS-1:0]                            per_bank_dram_wb_req,
    input  wire[`NUMBER_BANKS-1:0][31:0]                      per_bank_dram_wb_req_addr,
    input  wire[`NUMBER_BANKS-1:0][`BANK_LINE_SIZE_RNG][31:0] per_bank_dram_wb_req_data,

    // real Dram request
    output wire                                             dram_req,
    output wire                                             dram_req_write,
    output wire                                             dram_req_read,
    output wire [31:0]                                      dram_req_addr,
    output wire [31:0]                                      dram_req_size,
    output wire [`BANK_LINE_SIZE_RNG][31:0]                 dram_req_data
	
);


	wire dfqq_req;
	wire[31:0] dfqq_req_addr;
	wire dfqq_empty;
	wire dfqq_pop  = !dwb_valid && dfqq_req; // If no dwb, and dfqq has valids, then pop
	wire dfqq_push = (|per_bank_dram_fill_req);
	VX_cache_dfq_queue VX_cache_dfq_queue(
		.clk                        (clk),
		.reset                      (reset),
		.dfqq_push                  (dfqq_push),
		.per_bank_dram_fill_req     (per_bank_dram_fill_req),
		.per_bank_dram_fill_req_addr(per_bank_dram_fill_req_addr),
		.dfqq_pop                   (dfqq_pop),
		.dfqq_req                   (dfqq_req),
		.dfqq_req_addr              (dfqq_req_addr),
		.dfqq_empty                 (dfqq_empty),
		.dfqq_full                  (dfqq_full)
		);


	wire                              dwb_valid;
	wire[`vx_clog2(`NUMBER_BANKS)-1:0] dwb_bank;
	VX_generic_priority_encoder #(.N(`NUMBER_BANKS)) VX_sel_dwb(
		.valids(per_bank_dram_wb_req),
		.index (dwb_bank),
		.found (dwb_valid)
		);


	assign per_bank_dram_wb_queue_pop = per_bank_dram_wb_req & (~(1 << dwb_bank));


	assign dram_req       = dwb_valid || dfqq_req;
	assign dram_req_write = dwb_valid;
	assign dram_req_read  = dfqq_req && !dwb_valid;
	assign dram_req_addr  = (dwb_valid ? per_bank_dram_wb_req_addr[dwb_bank] : dfqq_req_addr) & `BASE_ADDR_MASK;
	assign dram_req_size  = `BANK_LINE_SIZE_BYTES;
	assign dram_req_data  = dwb_valid ? per_bank_dram_wb_req_data[dwb_bank] : 0;

endmodule