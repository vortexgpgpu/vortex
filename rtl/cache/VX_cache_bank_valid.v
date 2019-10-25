`include "../VX_define.v"

module VX_cache_bank_valid
#(
	parameter NUMBER_BANKS = 0
)
(
	input  wire [`NT_M1:0]                       i_p_valid,
	input  wire [`NT_M1:0][31:0]                 i_p_addr,
	output reg [NUMBER_BANKS - 1 : 0][`NT_M1:0]  thread_track_banks
);

	genvar t_id;
	always @(*) begin
		thread_track_banks = 0;
	    for (t_id = 0; t_id <= `NT_M1; t_id = t_id + 1)
	    begin
	    	thread_track_banks[i_p_addr[t_id][4:2]][t_id] = i_p_valid[t_id];
	    end
	end

endmodule