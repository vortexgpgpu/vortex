
`include "VX_cache_config.v"

module VX_fill_invalidator (
	input  wire       clk,
	input  wire       reset,

	input  wire       possible_fill,
	input  wire       success_fill,

	input  wire[31:0] fill_addr,

	output reg invalidate_fill
	
);


	`ifndef FILL_INVALIDATOR_ACTIVE

		assign invalidate_fill = 0;

	`else 

		reg[`FILL_INVALIDAOR_SIZE-1:0]         fills_active;
		reg[`FILL_INVALIDAOR_SIZE-1:0][31:0]   fills_address;


		reg                                         success_found;
		reg[(`vx_clog2(`FILL_INVALIDAOR_SIZE))-1:0] success_index;

		integer curr_fill;
		always @(*) begin
			assign invalidate_fill = 0;
			assign success_found   = 0;
			assign success_index   = 0;
			for (curr_fill = 0; curr_fill < `FILL_INVALIDAOR_SIZE; curr_fill=curr_fill+1) begin

				if (fill_addr[31:`LINE_SELECT_ADDR_START] == fills_address[curr_fill][31:`LINE_SELECT_ADDR_START]) begin
					if (possible_fill && fills_active[curr_fill]) begin
						assign invalidate_fill = 1;
					end

					if (success_fill) begin
						assign success_found = 1;
						assign success_index = curr_fill;
					end
				end
			end
		end




	wire [(`vx_clog2(`FILL_INVALIDAOR_SIZE))-1:0] enqueue_index;
	wire                                          enqueue_found;

	VX_generic_priority_encoder #(.N(`FILL_INVALIDAOR_SIZE)) VX_sel_bank(
		.valids(fills_active),
		.index (enqueue_index),
		.found (enqueue_found)
		);


	reg[`FILL_INVALIDAOR_SIZE-1:0] new_valids;



	always @(posedge clk) begin
		if (reset) begin
			fills_active  <= 0;
			fills_address <= 0;
		end else begin
			if (enqueue_found && !invalidate_fill) begin
				fills_active[enqueue_index]  <= 1;
				fills_address[enqueue_index] <= fill_addr;
			end

			if (success_found) begin
				fills_active[success_index] <= 0;
			end

		end
	end


	`endif


endmodule