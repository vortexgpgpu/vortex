`include "VX_cache_config.v"


module VX_cache_wb_sel_merge (

	// Per Bank WB
    input  wire [`NUMBER_BANKS-1:0][`NUMBER_REQUESTS-1:0] per_bank_wb_tid,
    input  wire [`NUMBER_BANKS-1:0][4:0]                  per_bank_wb_rd,
    input  wire [`NUMBER_BANKS-1:0][1:0]                  per_bank_wb_wb,
    input  wire [`NUMBER_BANKS-1:0][`NW_M1:0]             per_bank_wb_warp_num,
    input  wire [`NUMBER_BANKS-1:0][31:0]                 per_bank_wb_data,
    output wire [`NUMBER_BANKS-1:0]                       per_bank_wb_pop,


    // Core Writeback
    input  wire                              core_no_wb_slot,
    output reg  [`NUMBER_REQUESTS-1:0]       core_wb_valid,
    output reg  [`NUMBER_REQUESTS-1:0][31:0] core_wb_readdata
    output wire [4:0]                        core_wb_req_rd,
    output wire [1:0]                        core_wb_req_wb,
    output wire [`NW_M1:0]                   core_wb_warp_num,
	
);

	wire [`NUMBER_BANKS-1:0] per_bank_wb_pop_unqual;
	assign per_bank_wb_pop = per_bank_wb_pop_unqual & {`NUMBER_BANKS{core_no_wb_slot}};

	wire[`NUMBER_BANKS-1:0] bank_wants_wb;
	generate
		integer curr_bank;
		for (curr_bank = 0; curr_bank < `NUMBER_BANKS; curr_bank=curr_bank+1) begin
			assign bank_wants_wb[curr_bank] = (|per_bank_wb_valid[curr_bank]);
		end
	endgenerate


	wire [(`vx_clog2(`NUMBER_BANKS))-1:0] main_bank_index;
	wire                                  found_bank;

	VX_generic_priority_encoder #(.N(`NUMBER_BANKS)) VX_sel_bank(
		.valids(bank_wants_wb),
		.index (main_bank_index),
		.found (found_bank)
		);

	assign core_wb_req_rd   = per_bank_wb_rd      [main_bank_index];
	assign core_wb_req_wb   = per_bank_wb_wb      [main_bank_index];
	assign core_wb_warp_num = per_bank_wb_warp_num[main_bank_index];

	generate
		integer this_bank;
		for (this_bank = 0; this_bank < `NUMBER_BANKS; this_bank = this_bank + 1) begin
			if ((per_bank_wb_rd[this_bank] == per_bank_wb_rd[main_bank_index]) 
			            && (per_bank_wb_rd[this_bank] == per_bank_wb_rd[main_bank_index])) begin

				assign core_wb_valid[per_bank_wb_tid[this_bank]]    = 1;
				assign core_wb_readdata[per_bank_wb_tid[this_bank]] = per_bank_wb_data[this_bank];
				assign per_bank_wb_pop_unqual[this_bank]            = 1;
			end else
				assign per_bank_wb_pop_unqual[this_bank]            = 0;
			end
		end
	endgenerate

endmodule