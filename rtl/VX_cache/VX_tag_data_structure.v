module VX_tag_data_structure (
	input  wire        clk,

	input  wire[31:0]                      readaddr,
	output wire[`BANK_LINE_SIZE_RNG][31:0] readdata,

	input  wire[`BANK_LINE_SIZE_RNG][3]    writeenable,
	input  wire[31:0]                      writeaddr,
	input  wire[`BANK_LINE_SIZE_RNG][31:0] writedata
	
);

endmodule

// OFFSET_SIZE_RNG