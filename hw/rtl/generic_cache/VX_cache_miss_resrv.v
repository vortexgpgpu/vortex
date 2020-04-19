
`include "VX_cache_config.vh"

module VX_cache_miss_resrv #(
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
	input wire clk,
	input wire reset,

	// Miss enqueue
	input wire                                   miss_add,
	input wire[31:0]                             miss_add_addr,
	input wire[`WORD_SIZE_RNG]                   miss_add_data,
	input wire[`LOG2UP(NUM_REQUESTS)-1:0]        miss_add_tid,
	input wire[4:0]                              miss_add_rd,
	input wire[1:0]                              miss_add_wb,
	input wire[`NW_BITS-1:0]                     miss_add_warp_num,
	input wire[2:0]                              miss_add_mem_read,
	input wire[2:0]                              miss_add_mem_write,
	input wire[31:0]                             miss_add_pc,
	output wire                                  miss_resrv_full,
	output wire                                  miss_resrv_stop,

	// Broadcast Fill
	input wire                                   is_fill_st1,

/* verilator lint_off UNUSED */
    // TODO: should fix this
	input wire[31:0]                             fill_addr_st1,
/* verilator lint_on UNUSED */

	// Miss dequeue
	input  wire                                  miss_resrv_pop,
	output wire                                  miss_resrv_valid_st0,
	output wire[31:0]                            miss_resrv_addr_st0,
	output wire[`WORD_SIZE_RNG]                  miss_resrv_data_st0,
	output wire[`LOG2UP(NUM_REQUESTS)-1:0]       miss_resrv_tid_st0,
	output wire[4:0]                             miss_resrv_rd_st0,
	output wire[1:0]                             miss_resrv_wb_st0,
	output wire[`NW_BITS-1:0]                    miss_resrv_warp_num_st0,
	output wire[2:0]                             miss_resrv_mem_read_st0,
	output wire[31:0]                            miss_resrv_pc_st0,
	output wire[2:0]                             miss_resrv_mem_write_st0
	
);
	// Size of metadata = 32 + `LOG2UP(NUM_REQUESTS) + 5 + 2 + (`NW_BITS-1 + 1)
	reg [`MRVQ_METADATA_SIZE-1:0]   metadata_table[MRVQ_SIZE-1:0];
	reg [MRVQ_SIZE-1:0][31:0]       addr_table;
	reg [MRVQ_SIZE-1:0][31:0]       pc_table;
	reg [MRVQ_SIZE-1:0]             valid_table;
	reg [MRVQ_SIZE-1:0]             ready_table;
	reg [`LOG2UP(MRVQ_SIZE)-1:0]    head_ptr;
	reg [`LOG2UP(MRVQ_SIZE)-1:0]    tail_ptr;

	reg [31:0] size;

	// assign miss_resrv_full = (MRVQ_SIZE != 2) && (tail_ptr+1) == head_ptr;
	assign miss_resrv_full = (MRVQ_SIZE != 2) && (size ==  MRVQ_SIZE   );
	assign miss_resrv_stop = (MRVQ_SIZE != 2) && (size  > (MRVQ_SIZE-5));

	wire                           enqueue_possible = !miss_resrv_full;
	wire [`LOG2UP(MRVQ_SIZE)-1:0]  enqueue_index    = tail_ptr;

	reg [MRVQ_SIZE-1:0] make_ready;
	genvar curr_e;
	generate
		for (curr_e = 0; curr_e < MRVQ_SIZE; curr_e=curr_e+1) begin
			assign make_ready[curr_e] = is_fill_st1 && valid_table[curr_e]
													&& addr_table[curr_e][31:`LINE_SELECT_ADDR_START] == fill_addr_st1[31:`LINE_SELECT_ADDR_START];
		end
	endgenerate

	wire                          dequeue_possible = valid_table[head_ptr] && ready_table[head_ptr];
	wire [`LOG2UP(MRVQ_SIZE)-1:0] dequeue_index    = head_ptr;

	assign miss_resrv_valid_st0 = (MRVQ_SIZE != 2) && dequeue_possible;
	assign miss_resrv_pc_st0    = pc_table[dequeue_index];
	assign miss_resrv_addr_st0  = addr_table[dequeue_index];
	assign {miss_resrv_data_st0, miss_resrv_tid_st0, miss_resrv_rd_st0, miss_resrv_wb_st0, miss_resrv_warp_num_st0, miss_resrv_mem_read_st0, miss_resrv_mem_write_st0} = metadata_table[dequeue_index];

	wire mrvq_push = miss_add && enqueue_possible && (MRVQ_SIZE != 2);
	wire mrvq_pop  = miss_resrv_pop && dequeue_possible;

	wire update_ready = (|make_ready);
	integer i;
	always @(posedge clk) begin
		if (reset) begin
			for (i = 0; i < MRVQ_SIZE; i=i+1) begin
				metadata_table[i] <= 0;
			end
			valid_table <= 0;
			ready_table <= 0;
			addr_table  <= 0;
			pc_table    <= 0;
			size        <= 0;
			head_ptr    <= 0;
			tail_ptr    <= 0;
		end else begin
			if (mrvq_push) begin
				valid_table[enqueue_index]    <= 1;
				ready_table[enqueue_index]    <= 0;
				pc_table[enqueue_index]       <= miss_add_pc;
				addr_table[enqueue_index]     <= miss_add_addr;
				metadata_table[enqueue_index] <= {miss_add_data, miss_add_tid, miss_add_rd, miss_add_wb, miss_add_warp_num, miss_add_mem_read, miss_add_mem_write};
				tail_ptr                      <= tail_ptr + 1;
			end

			if (update_ready) begin
				ready_table <= ready_table | make_ready;
			end

			if (mrvq_pop) begin
				valid_table[dequeue_index]    <= 0;
				ready_table[dequeue_index]    <= 0;
				addr_table[dequeue_index]     <= 0;
				metadata_table[dequeue_index] <= 0;
				pc_table[dequeue_index]       <= 0;
				head_ptr                      <= head_ptr + 1;
			end

			if (!(mrvq_push && mrvq_pop)) begin
				if (mrvq_push) begin
					size <= size + 1;
				end

				if (mrvq_pop) begin
					size <= size - 1;
				end
			end
		end
	end

endmodule