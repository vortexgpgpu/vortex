`include "../VX_define.v"

module VX_cache_bank_valid
#(
	parameter NUMBER_BANKS = 8,
	parameter LOG_NUM_BANKS = 3,
	parameter NUM_REQ = 1
)
(
	input  wire [NUM_REQ-1:0]                       i_p_valid,
	input  wire [NUM_REQ-1:0][31:0]                 i_p_addr,
	output reg [NUMBER_BANKS - 1 : 0][NUM_REQ-1:0]  thread_track_banks
);

	generate
	integer t_id;
		always @(*) begin
			thread_track_banks = 0;
		    for (t_id = 0; t_id < NUM_REQ; t_id = t_id + 1)
		    begin
		    	thread_track_banks[i_p_addr[t_id][2+LOG_NUM_BANKS-1:2]][t_id] = i_p_valid[t_id];
		    end
		end
	endgenerate

endmodule
