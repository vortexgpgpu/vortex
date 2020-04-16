`include "VX_cache_config.vh"

module VX_prefetcher
	#(
		parameter PRFQ_SIZE                     = 64,
		parameter PRFQ_STRIDE                   = 2,
		// Size of line inside a bank in bytes
		parameter BANK_LINE_SIZE_BYTES          = 16,
		// Size of a word in bytes
		parameter WORD_SIZE_BYTES               = 4
	)
	(
	input wire  clk,
	input wire  reset,

	input wire       dram_req,
	input wire[31:0] dram_req_addr,

	input  wire       pref_pop,
	output wire       pref_valid,
	output wire[31:0] pref_addr
	
);


	reg[`LOG2UP(PRFQ_STRIDE):0]   use_valid;
	reg[31:0]                     use_addr;


	wire       current_valid;
	wire[31:0] current_addr;

	wire current_full;
	wire current_empty;


	assign current_valid = ~current_empty;

	wire update_use = ((use_valid == 0) || ((use_valid-1) == 0)) && current_valid;

	VX_generic_queue_ll #(.DATAW(32), .SIZE(PRFQ_SIZE)) pfq_queue(
		.clk     (clk),
		.reset   (reset),

		.push    (dram_req && !current_full && !pref_pop),
		.in_data (dram_req_addr & `BASE_ADDR_MASK),

		.pop     (update_use),
		.out_data(current_addr),

		.empty   (current_empty),
		.full    (current_full)
		);



	assign pref_valid = use_valid != 0;
	assign pref_addr  = use_addr;


	always @(posedge clk) begin
		if (reset) begin
			use_valid <= 0;
			use_addr  <= 0;
		end else begin 
			if (update_use) begin
				use_valid <= PRFQ_STRIDE;
				use_addr  <= current_addr + BANK_LINE_SIZE_BYTES;
			end else if (pref_valid && pref_pop) begin
				use_valid <= use_valid - 1;
				use_addr  <= use_addr + BANK_LINE_SIZE_BYTES;
			end

		end
	end

endmodule