module VX_lsu_addr_gen (
	input  wire[`NT_M1:0][31:0] base_address,
	input  wire[31:0]           offset,
	output wire[`NT_M1:0][31:0] address
	
);


	genvar index;
	for (index = 0; index < `NT; index = index + 1)
	begin
		assign address[index] = base_address[index] + offset;
	end

endmodule