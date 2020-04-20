`include "VX_define.v"

module VX_lsu_addr_gen (
	input  wire[`NUM_THREADS-1:0][31:0] base_address,
	input  wire[31:0]           offset,
	output wire[`NUM_THREADS-1:0][31:0] address
	
);
	genvar i;
	generate
	for (i = 0; i < `NUM_THREADS; i = i + 1) begin : addresses
		assign address[i] = base_address[i] + offset;
	end
	endgenerate

endmodule