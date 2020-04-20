`include "../VX_define.vh"

module VX_priority_encoder_sm
	#(
		parameter NB            = 4,
		parameter BITS_PER_BANK = 3,
		parameter NUM_REQ       = 3
	)
	(
	//INPUTS
	input wire			        	clk,
	input wire				        reset,
	input wire[`NUM_THREADS-1:0]			in_valid,
	input wire[`NUM_THREADS-1:0][31:0] 		in_address,
	input wire[`NUM_THREADS-1:0][31:0] 		in_data,
	// OUTPUTS
	// To SM Module
	output reg[NB:0]         	    out_valid,
	output reg[NB:0][31:0]  	    out_address,
	output reg[NB:0][31:0] 		    out_data,

	// To Processor
	output wire[NB:0][`LOG2UP(NUM_REQ) - 1:0]		    req_num,
	output reg 			            stall,
	output wire                     send_data // Finished all of the requests
);

	reg[`NUM_THREADS-1:0] left_requests;
	reg[`NUM_THREADS-1:0] serviced;

	wire[`NUM_THREADS-1:0] use_valid;

	wire requests_left = (|left_requests);

	assign use_valid = (requests_left) ? left_requests : in_valid;

	wire[NB:0][`NUM_THREADS-1:0] bank_valids;

	VX_bank_valids #(
		.NB(NB), 
		.BITS_PER_BANK(BITS_PER_BANK)
	) bank_valid (
		.valids_i(use_valid),
		.addr_i(in_address),
		.bank_valids(bank_valids)
	);

	wire[NB:0] more_than_one_valid;

	genvar curr_bank;
	generate
		for (curr_bank = 0; curr_bank <= NB; curr_bank = curr_bank + 1) begin : countones_blocks
			wire[`LOG2UP(`NUM_THREADS):0] num_valids;

			VX_countones #(.N(`NUM_THREADS)) valids_counter (
				.valids(bank_valids[curr_bank]),
				.count (num_valids)
				);
			assign more_than_one_valid[curr_bank] = num_valids > 1;
			// assign more_than_one_valid[curr_bank] = $countones(bank_valids[curr_bank]) > 1;
		end
	endgenerate


	assign stall     = (|more_than_one_valid);
	assign send_data = (!stall) && (|in_valid); // change

	wire[NB:0][(`LOG2UP(NUM_REQ)) - 1:0] internal_req_num;
	wire[NB:0]      internal_out_valid;


	// There's one or less valid per bank
	genvar curr_bank_o;
	generate
	for (curr_bank_o = 0; curr_bank_o <= NB; curr_bank_o = curr_bank_o + 1) begin : encoders

		VX_generic_priority_encoder #(
			.N(NUM_REQ)
		) priority_encoder (
		    .valids(bank_valids[curr_bank_o]),
		    .index(internal_req_num[curr_bank_o]),
		    .found(internal_out_valid[curr_bank_o])
		);
		assign out_address[curr_bank_o] = internal_out_valid[curr_bank_o] ? in_address[internal_req_num[curr_bank_o]] : 0;
		assign out_data[curr_bank_o]    = internal_out_valid[curr_bank_o] ? in_data[internal_req_num[curr_bank_o]] : 0;
	end
	endgenerate

	integer curr_b;
	always @(*) begin
		serviced = 0;
		for (curr_b = 0; curr_b <= NB; curr_b=curr_b+1) begin
			serviced[internal_req_num[curr_b]] = 1;
		end
	end

	assign req_num   = internal_req_num;
	assign out_valid = internal_out_valid;

	wire[`NUM_THREADS-1:0] serviced_qual = in_valid & (serviced);

	wire[`NUM_THREADS-1:0] new_left_requests = (left_requests == 0) ? (in_valid & ~serviced_qual) : (left_requests & ~ serviced_qual);

	// wire[`NUM_THREADS-1:0] new_left_requests = left_requests & ~(serviced_qual);

	always @(posedge clk) begin
		if (reset) begin
			left_requests <= 0;
			// serviced       = 0;
		end else begin
			if (!stall)    left_requests <= 0;
			else           left_requests <= new_left_requests;
		end
	end

endmodule