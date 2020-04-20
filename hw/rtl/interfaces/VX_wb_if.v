`ifndef VX_WB_INTER
`define VX_WB_INTER

`include "../VX_define.v"

interface VX_wb_if ();

	wire [`NUM_THREADS-1:0][31:0] 	write_data; 
	wire [31:0]            			wb_pc;
	wire [4:0]      		  		rd;
	wire [1:0]      		  		wb;
	wire [`NUM_THREADS-1:0]        	wb_valid;
	wire [`NW_BITS-1:0]        		wb_warp_num;

endinterface

`endif