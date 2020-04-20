module VX_generic_register #( 
	parameter N, 
	parameter PassThru = 0
) (
`IGNORE_WARNINGS_BEGIN
	input wire          clk,
	input wire          reset,
	input wire          stall,
	input wire          flush,
`IGNORE_WARNINGS_END
	input wire[N-1:0]  	in,
	output wire[N-1:0] 	out
);

	if (PassThru) begin
		assign out = in;
	end else begin

		reg [(N-1):0] value;

		always @(posedge clk) begin
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