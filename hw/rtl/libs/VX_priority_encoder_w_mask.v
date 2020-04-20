`include "VX_define.v"
module VX_priority_encoder_w_mask #(
	parameter N = 10
)  (
    input  wire[N-1:0]        		valids,
    output reg [N-1:0]        		mask,
    //output reg[$clog2(N)-1:0] index,
    output reg[(`LOG2UP(N))-1:0] 	index,
    //output reg[`LOG2UP(N):0] index, // eh
    output reg                		found
);

	integer i;
	always @(valids) begin
		index = 0;
		found = 0;
		// mask  = 0;
		for (i = 0; i < N; i=i+1) begin
			if (valids[i]) begin
				//index = i[$clog2(N)-1:0];
        		index = i[(`LOG2UP(N))-1:0];
				found = 1;
				// mask[index] = (1 << i);
				// $display("%h",(1 << i));
			end
		end
	end

	assign mask = found ? (1 << index) : 0;
endmodule