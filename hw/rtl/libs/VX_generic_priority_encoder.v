`ifndef VX_GENERIC_PRIORITY_ENCODER
`define VX_GENERIC_PRIORITY_ENCODER

`include "VX_define.vh"

module VX_generic_priority_encoder
  #(
  	parameter N = 1
  )
  (
    input  wire[N-1:0] valids,
    //output reg[$clog2(N)-1:0] index,
    output reg[(`LOG2UP(N))-1:0] index,
    //output reg[`LOG2UP(N):0] index, // eh
    output reg           found
  );

	integer i;
	always @(*) begin
		index = 0;
		found = 0;
		for (i = N-1; i >= 0; i = i - 1) begin
			if (valids[i]) begin
				//index = i[$clog2(N)-1:0];
        		index = i[(`LOG2UP(N))-1:0];
				found = 1;
			end
		end
	end
endmodule

`endif