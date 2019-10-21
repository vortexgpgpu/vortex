module VX_generic_stack
	#(
		parameter WIDTH = 40,
		parameter DEPTH = 2
	)
	(
	input  wire              clk,
	input  wire              reset,
	input  wire              push,
	input  wire              pop,
	input  wire[WIDTH - 1:0] d,
	output reg [WIDTH - 1:0] q,
	);


	reg [DEPTH - 1:0] ptr;
	reg [WIDTH - 1:0] stack [0:(1 << DEPTH) - 1];

	always @(posedge clk) begin
		if (reset)
			ptr <= 0;
		else if (push)
			ptr <= ptr + 1;
		else if (pop)
			ptr <= ptr - 1;
	end

	always @(posedge clk) begin
		if (push) begin
			if(push)
				stack[ptr] <= q;
		end
	end

	always @(*) begin
		if (pop)
			q <= stack[ptr - 1];
	end

endmodule