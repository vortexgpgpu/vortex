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
	input  reg [WIDTH - 1:0] q1,
	input  reg [WIDTH - 1:0] q2,
	output wire[WIDTH - 1:0] d
	);


	reg [DEPTH - 1:0] ptr;
	reg [WIDTH - 1:0] stack [0:(1 << DEPTH) - 1];

	integer i;
	always @(posedge clk) begin
		if (reset) begin
			ptr <= 0;
			for (i = 0; i < (1 << DEPTH); i=i+1) stack[i] <= 0;
		end else 
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