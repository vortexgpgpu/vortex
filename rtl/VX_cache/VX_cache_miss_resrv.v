
`include "VX_cache_config.v"

module VX_cache_miss_resrv (
	input wire clk,
	input wire reset,

	// Miss enqueue
	input wire                                   miss_add,
	input wire[31:0]                             miss_add_addr,
	input wire[31:0]                             miss_add_data,
	input wire[`vx_clog2(`NUMBER_REQUESTS)-1:0]  miss_add_tid,
	input wire[4:0]                              miss_add_rd,
	input wire[1:0]                              miss_add_wb,
	input wire[`NW_M1:0]                         miss_add_warp_num,
	input wire[2:0]                              miss_add_mem_read,
	input wire[2:0]                              miss_add_mem_write,
	output wire                                  miss_resrv_full,

	// Broadcast Fill
	input wire                                   is_fill_st1,
	input wire[31:0]                             fill_addr_st1,

	// Miss dequeue
	input  wire                                  miss_resrv_pop,
	output wire                                  miss_resrv_valid_st0,
	output wire[31:0]                            miss_resrv_addr_st0,
	output wire[31:0]                            miss_resrv_data_st0,
	output wire[`vx_clog2(`NUMBER_REQUESTS)-1:0] miss_resrv_tid_st0,
	output wire[4:0]                             miss_resrv_rd_st0,
	output wire[1:0]                             miss_resrv_wb_st0,
	output wire[`NW_M1:0]                        miss_resrv_warp_num_st0,
	output wire[2:0]                             miss_resrv_mem_read_st0,
	output wire[2:0]                             miss_resrv_mem_write_st0
	
);

	// Size of metadata = 32 + `vx_clog2(`NUMBER_REQUESTS) + 5 + 2 + (`NW_M1 + 1)
	reg[`MRVQ_METADATA_SIZE-1:0] metadata_table[`MRVQ_SIZE-1:0];
	reg[`MRVQ_SIZE-1:0][31:0]    addr_table;
	reg[`MRVQ_SIZE-1:0]          valid_table;
	reg[`MRVQ_SIZE-1:0]          ready_table;


	assign miss_resrv_full = !(&valid_table);


	wire                            enqueue_possible;
	wire[`vx_clog2(`MRVQ_SIZE)-1:0] enqueue_index;
	VX_generic_priority_encoder #(.N(`MRVQ_SIZE)) enqueue_picker(
		.valids(~valid_table),
		.index (enqueue_index),
		.found (enqueue_possible)
		);

	reg[`MRVQ_SIZE-1:0] make_ready;
	genvar curr_e;
	generate
		for (curr_e = 0; curr_e < `MRVQ_SIZE; curr_e=curr_e+1) begin
			assign make_ready[curr_e] = is_fill_st1 && valid_table[curr_e]
			                                        && addr_table[curr_e][31:`LINE_SELECT_ADDR_START] == fill_addr_st1[31:`LINE_SELECT_ADDR_START];
		end
	endgenerate

	wire                            dequeue_possible;
	wire[`vx_clog2(`MRVQ_SIZE)-1:0] dequeue_index;
	wire[`MRVQ_SIZE-1:0]            dequeue_valid = valid_table & ready_table;
	VX_generic_priority_encoder #(.N(`MRVQ_SIZE)) dequeue_picker(
		.valids(dequeue_valid),
		.index (dequeue_index),
		.found (dequeue_possible)
		);

	assign miss_resrv_valid_st0 = dequeue_possible;
	assign miss_resrv_addr_st0  = addr_table[dequeue_index];
	assign {miss_resrv_data_st0, miss_resrv_tid_st0, miss_resrv_rd_st0, miss_resrv_wb_st0, miss_resrv_warp_num_st0, miss_resrv_mem_read_st0, miss_resrv_mem_write_st0} = metadata_table[dequeue_index];

	wire update_ready = (|make_ready);
	integer i;
	always @(posedge clk or reset) begin
		if (reset) begin
			for (i = 0; i < `MRVQ_SIZE; i=i+1) metadata_table[i] <= 0;
			valid_table <= 0;
			ready_table <= 0;
			addr_table  <= 0;
		end else begin
			if (miss_add && enqueue_possible) begin
				valid_table[enqueue_index]    <= 1;
				ready_table[enqueue_index]    <= 0;
				addr_table[enqueue_index]     <= miss_add_addr;
				metadata_table[enqueue_index] <= {miss_add_data, miss_add_tid, miss_add_rd, miss_add_wb, miss_add_warp_num, miss_add_mem_read, miss_add_mem_write};
			end

			if (update_ready) begin
				ready_table = ready_table | make_ready;
			end

			if (miss_resrv_pop && dequeue_possible) begin
				valid_table[dequeue_index]    <= 0;
				ready_table[dequeue_index]    <= 0;
				addr_table[dequeue_index]     <= 0;
				metadata_table[dequeue_index] <= 0;
			end

		end
	end


endmodule