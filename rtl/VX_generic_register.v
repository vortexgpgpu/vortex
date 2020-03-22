
module VX_generic_register 
	#( parameter N = 1, parameter Valid = 1)
	(
		input wire           clk,
		input wire           reset,
		input wire           stall,
		input wire           flush,
		input wire[(N-1):0]  in,
		output wire[(N-1):0] out
	);

	if (Valid == 0) begin

		assign out = in;

	end else begin

		reg[(N-1):0] value;

		always @(posedge clk or posedge reset) begin
			if (reset) begin
				value <= 0;
			end else if (flush) begin
				value <= 0;
			end else if (~stall) begin
				value <= in;
			end
		end

		assign out = value;

	end

endmodule