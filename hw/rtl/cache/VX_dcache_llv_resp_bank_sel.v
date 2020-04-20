`include "VX_cache_config.vh"

module VX_dcache_llv_resp_bank_sel #(
	// Size of cache in bytes
	parameter CACHE_SIZE_BYTES              = 1024, 
	// Size of line inside a bank in bytes
	parameter BANK_LINE_SIZE_BYTES          = 16, 
	// Number of banks {1, 2, 4, 8,...}
	parameter NUM_BANKS                     = 8, 
	// Size of a word in bytes
	parameter WORD_SIZE_BYTES               = 4, 
	// Number of Word requests per cycle {1, 2, 4, 8, ...}
	parameter NUM_REQUESTS                  = 2, 
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
) (
    output reg [NUM_BANKS-1:0]                             per_bank_llvq_pop,
    input  wire[NUM_BANKS-1:0]                             per_bank_llvq_valid,
    input  wire[NUM_BANKS-1:0][31:0]                       per_bank_llvq_rsp_addr,
    input  wire[NUM_BANKS-1:0][`BANK_LINE_WORDS-1:0][31:0] per_bank_llvq_rsp_data,
 	input  wire[NUM_BANKS-1:0][`LOG2UP(NUM_REQUESTS)-1:0]  per_bank_llvq_rsp_tid,
 
   	input  wire                                              llvq_pop,
    output reg[NUM_REQUESTS-1:0]                             llvq_valid,
    output reg[NUM_REQUESTS-1:0][31:0]                       llvq_rsp_addr,
    output reg[NUM_REQUESTS-1:0][`BANK_LINE_WORDS-1:0][31:0] llvq_rsp_data
);

	wire [(`LOG2UP(NUM_BANKS))-1:0] main_bank_index;
	wire                            found_bank;

	VX_generic_priority_encoder #(
		.N(NUM_BANKS)
	) sel_bank(
		.valids(per_bank_llvq_valid),
		.index (main_bank_index),
		.found (found_bank)
	);

	always @(*) begin
		llvq_valid = 0;
		llvq_rsp_addr = 0;
		llvq_rsp_data = 0;
		per_bank_llvq_pop = 0;
		if (found_bank && llvq_pop) begin
			llvq_valid   [per_bank_llvq_rsp_tid[main_bank_index]] = 1'b1;
			llvq_rsp_addr[per_bank_llvq_rsp_tid[main_bank_index]] = per_bank_llvq_rsp_addr[main_bank_index];
			llvq_rsp_data[per_bank_llvq_rsp_tid[main_bank_index]] = per_bank_llvq_rsp_data[main_bank_index];
			per_bank_llvq_pop[main_bank_index]                    = 1'b1;
		end
	end

endmodule
