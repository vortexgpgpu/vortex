`include "VX_cache_config.v"

module VX_tag_data_access (
	input  wire                            clk,
	input  wire                            reset,

	input  wire                            valid_st10,    
	input  wire                            is_fill_st10,  
	input  wire[31:0]                      readaddr_st10, 
	input  wire[`BANK_LINE_SIZE_RNG][31:0] filldata_st10,

	input  wire[31:0]                      writeaddr_st1e,
	input  wire[31:0]                      writeword_st1e,
	input  wire[2:0]                       mem_write_st1e,
	input  wire[2:0]                       mem_read_st1e, 

	output wire[31:0]                      readword_st1e,
	output wire[`BANK_LINE_SIZE_RNG][31:0] readdata_st1e,
	output wire                            miss_st1e
	
);


	reg[`BANK_LINE_SIZE_RNG][31:0] readdata_st[`STAGE_1_CYCLES-1:0];

endmodule