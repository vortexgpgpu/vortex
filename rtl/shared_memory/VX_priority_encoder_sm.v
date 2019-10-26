`include "../VX_define.v"

module VX_priority_encoder_sm
	#(
		parameter NB            = 4,
		parameter BITS_PER_BANK = 3
	)
	(
	//INPUTS
	input wire				clk,
	//input wire				reset,
	input wire[`NT_M1:0]			in_valid,
	input wire[`NT_M1:0][31:0] 		in_address,
	input wire[`NT_M1:0][31:0] 		in_data,
	// OUTPUTS
	// To SM Module
	output reg[NB:0]         	    out_valid,
	output reg[NB:0][31:0]  	    out_address,
	output reg[NB:0][31:0] 		    out_data,

	// To Processor
	output wire[NB:0][1:0]		    req_num,
	output reg 			            stall,
	output wire                     send_data // Finished all of the requests
);

	reg[`NT_M1:0] left_requests;

	wire[`NT_M1:0] use_valid;


	wire requests_left = (|left_requests);

	assign use_valid = (requests_left) ? left_requests : in_valid;


	wire[NB:0][`NT_M1:0] bank_valids;
	VX_bank_valids #(.NB(NB), .BITS_PER_BANK(BITS_PER_BANK)) vx_bank_valid(
		.in_valids(use_valid),
		.in_addr(in_address),
		.bank_valids(bank_valids)
		);

	wire[NB:0] more_than_one_valid;

	genvar curr_bank;
	for (curr_bank = 0; curr_bank <= NB; curr_bank = curr_bank + 1) 
	begin
		assign more_than_one_valid[curr_bank] = $countones(bank_valids[curr_bank]) > 1;
	end


	assign stall     = (|more_than_one_valid);
	assign send_data = (!stall) && (|in_valid); // change

	wire[NB:0][1:0] internal_req_num;
	wire[NB:0]      internal_out_valid;


	// There's one or less valid per bank
	genvar curr_bank_o;
	for (curr_bank_o = 0; curr_bank_o <= NB; curr_bank_o = curr_bank_o + 1) 
	begin

		VX_generic_priority_encoder #(.N(4)) vx_priority_encoder(
		    .valids(bank_valids[curr_bank_o]),
		    .index(internal_req_num[curr_bank_o]),
		    .found(internal_out_valid[curr_bank_o])
		  );
		assign out_address[curr_bank_o] = internal_out_valid[curr_bank_o] ? in_address[internal_req_num[curr_bank_o]] : 0;
		assign out_data[curr_bank_o]    = internal_out_valid[curr_bank_o] ? in_data[internal_req_num[curr_bank_o]] : 0;
	end

	reg[`NT_M1:0] serviced;
	integer curr_b;
	always @(*) begin
		serviced = 0;
		for (curr_b = 0; curr_b <= NB; curr_b=curr_b+1) begin
			serviced[internal_req_num[curr_b]] = 1;
		end
	end


	assign req_num   = internal_req_num;
	assign out_valid = internal_out_valid;


	wire[`NT_M1:0] serviced_qual = in_valid & (serviced);

	wire[`NT_M1:0] new_left_requests = (left_requests == 0) ? (in_valid & ~serviced_qual) : (left_requests & ~ serviced_qual);

	// wire[`NT_M1:0] new_left_requests = left_requests & ~(serviced_qual);

	always @(posedge clk) begin
		if (!stall)    left_requests <= 0;
		else           left_requests <= new_left_requests;
	end

endmodule