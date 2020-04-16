`include "VX_cache_config.vh"

module VX_cache_wb_sel_merge
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
    // Function ID, {Dcache=0, Icache=1, Sharedmemory=2}
    parameter FUNC_ID                       = 0,

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

	// Per Bank WB
	input  wire [NUM_BANKS-1:0]                                  per_bank_wb_valid,
    input  wire [NUM_BANKS-1:0][`LOG2UP(NUM_REQUESTS)-1:0]       per_bank_wb_tid,
    input  wire [NUM_BANKS-1:0][4:0]                             per_bank_wb_rd,
    input  wire [NUM_BANKS-1:0][1:0]                             per_bank_wb_wb,
    input  wire [NUM_BANKS-1:0][`NW_BITS-1:0]                    per_bank_wb_warp_num,
    input  wire [NUM_BANKS-1:0][`WORD_SIZE_RNG]                  per_bank_wb_data,
    input  wire [NUM_BANKS-1:0][31:0]                            per_bank_wb_pc,
    input  wire [NUM_BANKS-1:0][31:0]                            per_bank_wb_address,
    output wire [NUM_BANKS-1:0]                                  per_bank_wb_pop,


    // Core Writeback
    input  wire                                                   core_no_wb_slot,
    output reg  [NUM_REQUESTS-1:0]                                core_wb_valid,
    output reg  [NUM_REQUESTS-1:0][`WORD_SIZE_RNG]                core_wb_readdata,
    output reg  [NUM_REQUESTS-1:0][31:0]                          core_wb_pc,
    output wire [4:0]                                             core_wb_req_rd,
    output wire [1:0]                                             core_wb_req_wb,
    output wire [`NW_BITS-1:0]                                    core_wb_warp_num,
    output reg  [NUM_REQUESTS-1:0][31:0]                          core_wb_address
	
);

	reg [NUM_BANKS-1:0] per_bank_wb_pop_unqual;
	assign per_bank_wb_pop = per_bank_wb_pop_unqual & {NUM_BANKS{~core_no_wb_slot}};

	// wire[NUM_BANKS-1:0] bank_wants_wb;
	// genvar curr_bank;
	// generate
	// 	for (curr_bank = 0; curr_bank < NUM_BANKS; curr_bank=curr_bank+1) begin
	// 		assign bank_wants_wb[curr_bank] = (|per_bank_wb_valid[curr_bank]);
	// 	end
	// endgenerate


	wire [`LOG2UP(NUM_BANKS)-1:0] main_bank_index;
	wire                                  found_bank;

	VX_generic_priority_encoder #(.N(NUM_BANKS)) VX_sel_bank(
		.valids(per_bank_wb_valid),
		.index (main_bank_index),
		.found (found_bank)
		);

	assign core_wb_req_rd   = per_bank_wb_rd[main_bank_index];
	assign core_wb_req_wb   = per_bank_wb_wb[main_bank_index];
	assign core_wb_warp_num = per_bank_wb_warp_num[main_bank_index];

	integer this_bank;
	generate
		always @(*) begin
			core_wb_valid    = 0;
			core_wb_readdata = 0;
			core_wb_pc       = 0;
			core_wb_address  = 0;
			for (this_bank = 0; this_bank < NUM_BANKS; this_bank = this_bank + 1) begin
				if ((FUNC_ID == `L2FUNC_ID) || (FUNC_ID == `L3FUNC_ID)) begin

						if (found_bank
						 && !core_wb_valid[per_bank_wb_tid[this_bank]] 
						 && per_bank_wb_valid[this_bank] 
						 && ((main_bank_index == `LOG2UP(NUM_BANKS)'(this_bank)) 
						  || (per_bank_wb_tid[this_bank] != per_bank_wb_tid[main_bank_index]))) begin
							core_wb_valid[per_bank_wb_tid[this_bank]]    = 1;
							core_wb_readdata[per_bank_wb_tid[this_bank]] = per_bank_wb_data[this_bank];
							core_wb_pc[per_bank_wb_tid[this_bank]]       = per_bank_wb_pc[this_bank];
							core_wb_address[per_bank_wb_tid[this_bank]]  = per_bank_wb_address[this_bank];
							per_bank_wb_pop_unqual[this_bank]            = 1;
						end else begin
							per_bank_wb_pop_unqual[this_bank]            = 0;
						end

				end else begin

						if (((main_bank_index == `LOG2UP(NUM_BANKS)'(this_bank))
						  || (per_bank_wb_tid[this_bank] != per_bank_wb_tid[main_bank_index])) 
						&& found_bank 
						&& !core_wb_valid[per_bank_wb_tid[this_bank]] 
						&& (per_bank_wb_valid[this_bank]) 
						&& (per_bank_wb_rd[this_bank] == per_bank_wb_rd[main_bank_index]) 
						&& (per_bank_wb_warp_num[this_bank] == per_bank_wb_warp_num[main_bank_index])) begin
							core_wb_valid[per_bank_wb_tid[this_bank]]    = 1;
							core_wb_readdata[per_bank_wb_tid[this_bank]] = per_bank_wb_data[this_bank];
							core_wb_pc[per_bank_wb_tid[this_bank]]       = per_bank_wb_pc[this_bank];
							core_wb_address[per_bank_wb_tid[this_bank]]  = per_bank_wb_address[this_bank];
							per_bank_wb_pop_unqual[this_bank]            = 1;
						end else begin
							per_bank_wb_pop_unqual[this_bank]            = 0;

						end

				end

			end
		end
	endgenerate

endmodule