`include "VX_define.vh"

module VX_priority_encoder (
    input  wire[`NUM_WARPS-1:0] valids,
    output reg[`NW_BITS-1:0] 	index,
    output reg           		found
);

	integer i;
	always @(*) begin
		index = 0;
		found = 0;
		for (i = `NUM_WARPS-1; i >= 0; i = i - 1) begin
			if (valids[i]) begin
				index = i[`NW_BITS-1:0];
				found = 1;
			end
		end
	end
	
endmodule