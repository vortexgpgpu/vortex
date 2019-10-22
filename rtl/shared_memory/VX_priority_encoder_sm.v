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
	output reg[NB:0][31:0] 		out_data,

	// To Processor
	output wire[NB:0][1:0]		    req_num,
	output reg 			            stall,
	output wire                      send_data // Finished all of the requests
);

wire[NB:0][`NT_M1:0] bank_valids;
wire[NB:0][`NT_M1:0] temp_bank_valids;
reg[NB:0][`NT_M1:0] temp_valid; // State - If there's any ones here, then stall
wire[NB:0] temp_stall;
integer counter[NB:0] ;
wire[NB:0][`NT_M1:0] mask;
wire[NB:0] update_temp_valid; 
reg[NB:0] req_done;

VX_bank_valids #(.NB(NB), .BITS_PER_BANK(BITS_PER_BANK)) vx_bank_valid(
	.in_valids(in_valid),
	.in_addr(in_address),
	.bank_valids(bank_valids)
	);

genvar j;
for(j=0; j <= NB; j++) begin
	assign temp_stall[j] = ($countones(temp_valid[j]) != 0);
	assign temp_bank_valids[j] = (temp_stall[j] || req_done[j]) ? temp_valid[j] : bank_valids[j];
	assign update_temp_valid[j]  = !req_done[j] && ($countones(bank_valids[j]) > 1);

	VX_generic_priority_encoder #(.N(4)) vx_priority_encoder(
	    .valids(temp_bank_valids[j]),
	    .index(req_num[j]),
	    .found(out_valid[j])
	  );

	VX_set_bit vx_set_bit(
		.index(req_num[j]),
		.mask (mask[j])
	);

	assign out_address[j] = out_valid[j] ? in_address[req_num[j]] : 0;
	assign out_data[j]    = out_valid[j] ? in_data[req_num[j]] : 0;
end


assign stall = |temp_stall;
assign send_data = &req_done;

genvar i;
always @(posedge clk) begin
	for(i = 0; i <= NB; i = i+1) begin
		if (update_temp_valid[i]) begin
			counter[i] <= counter[i] + 1;
			if(counter[i] == 0) temp_valid[i] <= bank_valids[i] & mask[i];
			else if (counter[i] > 0) temp_valid[i] <= temp_bank_valids[i] & mask[i];
		end 
		if(($countones(in_valid) > 0) && ($countones(bank_valids[i]) == 0)) begin
			req_done[i] <= 1;
		end
		else if((counter[i][2:0] == ($countones(bank_valids[i])-1))) begin 
			req_done[i] <= 1;
			counter[i] <= 0;
		end
		else begin
			req_done[i] <= 0;
		end
	end
end

endmodule