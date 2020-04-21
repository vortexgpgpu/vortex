
`ifndef VX_MEM_WB_INST_INTER
`define VX_MEM_WB_INST_INTER

`include "VX_define.vh"

interface VX_inst_mem_wb_if ();

	wire [`NUM_THREADS-1:0][31:0]	loaded_data; 
	wire [31:0]            			mem_wb_pc;
	wire [4:0]      		  		rd;
	wire [1:0]      		  		wb;
	wire [`NUM_THREADS-1:0]        	wb_valid;
	wire [`NW_BITS-1:0]        		wb_warp_num;

endinterface

`endif