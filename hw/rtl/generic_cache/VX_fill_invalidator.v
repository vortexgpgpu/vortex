`include "VX_cache_config.vh"

module VX_fill_invalidator
	#(
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


	)
	(
	input  wire       clk,
	input  wire       reset,

	input  wire       possible_fill,
	input  wire       success_fill,

	input  wire[31:0] fill_addr,

	output reg invalidate_fill
	
);


	if (FILL_INVALIDAOR_SIZE == 0) begin

		assign invalidate_fill = 0;

	end else begin 

		reg[FILL_INVALIDAOR_SIZE-1:0]         fills_active;
		reg[FILL_INVALIDAOR_SIZE-1:0][31:0]   fills_address;


		reg[FILL_INVALIDAOR_SIZE-1:0] matched_fill;
		wire matched;
		integer fi;
		always @(*) begin
			for (fi = 0; fi < FILL_INVALIDAOR_SIZE; fi+=1) begin
				matched_fill[fi] = fills_active[fi] && (fills_address[fi][31:`LINE_SELECT_ADDR_START] == fill_addr[31:`LINE_SELECT_ADDR_START]);
			end
		end


		assign matched = (|(matched_fill));


		wire [(`LOG2UP(FILL_INVALIDAOR_SIZE))-1:0]  enqueue_index;
		wire                                        enqueue_found;

		VX_generic_priority_encoder #(
			.N(FILL_INVALIDAOR_SIZE)
		) sel_bank (
			.valids(~fills_active),
			.index (enqueue_index),
			.found (enqueue_found)
		);

		assign invalidate_fill = possible_fill && matched;

		always @(posedge clk) begin
			if (reset) begin
				fills_active  <= 0;
				fills_address <= 0;
			end else begin

				if (possible_fill && !matched && enqueue_found) begin
					fills_active [enqueue_index] <= 1;
					fills_address[enqueue_index] <= fill_addr;
				end else if (success_fill && matched) begin
					fills_active <= fills_active & (~matched_fill);
				end

			end
		end

		// reg                                         success_found;
		// reg[(`LOG2UP(FILL_INVALIDAOR_SIZE))-1:0]  success_index;

		// integer curr_fill;
		// always @(*) begin
		// 	invalidate_fill = 0;
		// 	success_found   = 0;
		// 	success_index   = 0;
		// 	for (curr_fill = 0; curr_fill < FILL_INVALIDAOR_SIZE; curr_fill=curr_fill+1) begin

		// 		if (fill_addr[31:`LINE_SELECT_ADDR_START] == fills_address[curr_fill][31:`LINE_SELECT_ADDR_START]) begin
		// 			if (possible_fill && fills_active[curr_fill]) begin
		// 				invalidate_fill = 1;
		// 			end

		// 			if (success_fill) begin
		// 				success_found = 1;
		// 				success_index = curr_fill;
		// 			end
		// 		end
		// 	end
		// end

		// wire [(`LOG2UP(FILL_INVALIDAOR_SIZE))-1:0] enqueue_index;
		// wire                                          enqueue_found;

		// VX_generic_priority_encoder #(.N(FILL_INVALIDAOR_SIZE)) sel_bank(
		// 	.valids(~fills_active),
		// 	.index (enqueue_index),
		// 	.found (enqueue_found)
		// 	);

		// always @(posedge clk) begin
		// 	if (reset) begin
		// 		fills_active  <= 0;
		// 		fills_address <= 0;
		// 	end else begin
		// 		if (possible_fill && !invalidate_fill) begin
		// 			fills_active[enqueue_index]  <= 1;
		// 			fills_address[enqueue_index] <= fill_addr;
		// 		end

		// 		if (success_found) begin
		// 			fills_active[success_index] <= 0;
		// 		end

		// 	end
		// end

	end

endmodule