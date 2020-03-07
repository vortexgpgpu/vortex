`include "VX_cache_config.v"

module VX_cache_dfq_queue
	#(
	// Size of cache in bytes
	parameter CACHE_SIZE_BYTES              = 1024, 
	// Size of line inside a bank in bytes
	parameter BANK_LINE_SIZE_BYTES          = 16, 
	// Number of banks {1, 2, 4, 8,...}
	parameter NUMBER_BANKS                  = 8, 
	// Size of a word in bytes
	parameter WORD_SIZE_BYTES               = 4, 
	// Number of Word requests per cycle {1, 2, 4, 8, ...}
	parameter NUMBER_REQUESTS               = 2, 
	// Number of cycles to complete stage 1 (read from memory)
	parameter STAGE_1_CYCLES                = 2, 

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

	// Core Request Queue Size
	parameter REQQ_SIZE                     = 8, 
	// Miss Reserv Queue Knob
	parameter MRVQ_SIZE                     = 8, 
	// Dram Fill Rsp Queue Size
	parameter DFPQ_SIZE                     = 2, 
	// Snoop Req Queue
	parameter SNRQ_SIZE                     = 8, 

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
	// Core Writeback Queue Size
	parameter CWBQ_SIZE                     = 8, 
	// Dram Writeback Queue Size
	parameter DWBQ_SIZE                     = 4, 
	// Dram Fill Req Queue Size
	parameter DFQQ_SIZE                     = 8, 
	// Lower Level Cache Hit Queue Size
	parameter LLVQ_SIZE                     = 16, 

 	// Fill Invalidator Size {Fill invalidator must be active}
 	parameter FILL_INVALIDAOR_SIZE          = 16, 

// Dram knobs
	parameter SIMULATED_DRAM_LATENCY_CYCLES = 10


	)
	(
	input  wire                          clk,
	input  wire                          reset,
	input  wire                          dfqq_push,
    input  wire[NUMBER_BANKS-1:0]       per_bank_dram_fill_req,
    input  wire[NUMBER_BANKS-1:0][31:0] per_bank_dram_fill_req_addr,

	input  wire            dfqq_pop,
    output wire            dfqq_req,
    output wire[31:0]      dfqq_req_addr,
	output wire            dfqq_empty,
	output wire            dfqq_full
);

    wire[NUMBER_BANKS-1:0]       out_per_bank_dram_fill_req;
    wire[NUMBER_BANKS-1:0][31:0] out_per_bank_dram_fill_req_addr;


    reg [NUMBER_BANKS-1:0]       use_per_bank_dram_fill_req;
    reg [NUMBER_BANKS-1:0][31:0] use_per_bank_dram_fill_req_addr;


    wire[NUMBER_BANKS-1:0]       qual_bank_dram_fill_req;
    wire[NUMBER_BANKS-1:0][31:0] qual_bank_dram_fill_req_addr;

    wire[NUMBER_BANKS-1:0]       updated_bank_dram_fill_req;


    wire o_empty;

	wire use_empty = !(|use_per_bank_dram_fill_req);
	wire out_empty = !(|out_per_bank_dram_fill_req) || o_empty;

	wire push_qual = dfqq_push && !dfqq_full;
	wire pop_qual  = dfqq_pop  && use_empty && !out_empty;
	VX_generic_queue_ll #(.DATAW(NUMBER_BANKS * (1+32)), .SIZE(DFQQ_SIZE)) dfqq_queue(
		.clk     (clk),
		.reset   (reset),
		.push    (push_qual),
		.in_data ({per_bank_dram_fill_req, per_bank_dram_fill_req_addr}),
		.pop     (pop_qual),
		.out_data({out_per_bank_dram_fill_req, out_per_bank_dram_fill_req_addr}),
		.empty   (o_empty),
		.full    (dfqq_full)
		);


	assign qual_bank_dram_fill_req      = use_empty ? out_per_bank_dram_fill_req      : use_per_bank_dram_fill_req; 
	assign qual_bank_dram_fill_req_addr = use_empty ? out_per_bank_dram_fill_req_addr : use_per_bank_dram_fill_req_addr;

	wire[`vx_clog2(NUMBER_BANKS)-1:0] qual_request_index;
	wire                               qual_has_request;
	VX_generic_priority_encoder #(.N(NUMBER_BANKS)) VX_sel_bank(
		.valids(qual_bank_dram_fill_req),
		.index (qual_request_index),
		.found (qual_has_request)
		);

	assign dfqq_empty    = !qual_has_request;
	assign dfqq_req      = qual_bank_dram_fill_req     [qual_request_index];
	assign dfqq_req_addr = qual_bank_dram_fill_req_addr[qual_request_index];

	assign updated_bank_dram_fill_req = qual_bank_dram_fill_req & (~(1 << qual_request_index));

	always @(posedge clk) begin
		if (reset) begin
			use_per_bank_dram_fill_req      <= 0;
			use_per_bank_dram_fill_req_addr <= 0;
		end else begin
			if (dfqq_pop && qual_has_request) begin
				use_per_bank_dram_fill_req      <= updated_bank_dram_fill_req;
				use_per_bank_dram_fill_req_addr <= qual_bank_dram_fill_req_addr;
			end
		end
	end


endmodule