

module VX_generic_register 
	#(
		parameter N = 1
	)
	(
		input          clk,
		input          reset,
		input          stall,
		input          flush,
		input[N-1:0]   in,
		output [N-1:0] out
	);


	reg[N-1:0] value;


	always @(posedge clk or posedge reset) begin
		if (reset || flush) begin
			value <= 0;
		end else if (~stall) begin
			value <= in;
		end
	end


	assign out = value;

endmodule