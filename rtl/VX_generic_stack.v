module VX_generic_stack
	#(
		parameter WIDTH = 40,
		parameter DEPTH = 2
	)
	(
	input  wire              clk,
	input  wire              push,
	input  wire              pop,
	input  reg [WIDTH - 1:0] q1,
	input  reg [WIDTH - 1:0] q2,
	output wire[WIDTH - 1:0] d
	);


	reg [DEPTH - 1:0] ptr;
	reg [WIDTH - 1:0] stack [0:(1 << DEPTH) - 1];

	always @(posedge clk) begin
		// if (reset)
		// 	ptr <= 0;
		// else 
		if (push)
			ptr <= ptr + 2;
		else if (pop)
			ptr <= ptr - 1;
	end

	always @(posedge clk) begin
		if (push) begin
			stack[ptr]   <= q1;
			stack[ptr+1] <= q2;
		end
	end

	assign d = stack[ptr - 1];

endmodule