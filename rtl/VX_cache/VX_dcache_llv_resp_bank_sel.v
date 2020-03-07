`include "VX_cache_config.v"

module VX_dcache_llv_resp_bank_sel (
    output reg [`NUMBER_BANKS-1:0]                                   per_bank_llvq_pop,
    input  wire[`NUMBER_BANKS-1:0]                                   per_bank_llvq_valid,
    input  wire[`NUMBER_BANKS-1:0][31:0]                             per_bank_llvq_res_addr,
    input  wire[`NUMBER_BANKS-1:0][`BANK_LINE_SIZE_RNG][31:0]        per_bank_llvq_res_data,
 	input  wire[`NUMBER_BANKS-1:0][`vx_clog2(`NUMBER_REQUESTS)-1:0]  per_bank_llvq_res_tid,
 
   	input  wire                             llvq_pop,
    output reg[`NUMBER_REQUESTS-1:0]       llvq_valid,
    output reg[`NUMBER_REQUESTS-1:0][31:0] llvq_res_addr,
    output reg[`NUMBER_REQUESTS-1:0][`BANK_LINE_SIZE_RNG][31:0] llvq_res_data


);

	wire [(`vx_clog2(`NUMBER_BANKS))-1:0] main_bank_index;
	wire                                  found_bank;

	VX_generic_priority_encoder #(.N(`NUMBER_BANKS)) VX_sel_bank(
	.valids(per_bank_llvq_valid),
	.index (main_bank_index),
	.found (found_bank)
	);


	always @(*) begin
		llvq_valid = 0;
		llvq_res_addr = 0;
		llvq_res_data = 0;
		per_bank_llvq_pop = 0;
		if (found_bank && llvq_pop) begin
			llvq_valid   [per_bank_llvq_res_tid] = 1;
			llvq_res_addr[per_bank_llvq_res_tid] = per_bank_llvq_res_addr[main_bank_index];
			llvq_res_data[per_bank_llvq_res_tid] = per_bank_llvq_res_data[main_bank_index];
			per_bank_llvq_pop[main_bank_index]   = 1;
		end
	end

endmodule
