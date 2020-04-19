
`ifndef VX_EXEC_UNIT_WB_INST_INTER
`define VX_EXEC_UNIT_WB_INST_INTER

`include "../VX_define.vh"

interface VX_inst_exec_wb_if ();

	wire [`NUM_THREADS-1:0][31:0]  	alu_result; 
	wire [31:0]            			exec_wb_pc;
	wire [4:0]      		  		rd;
	wire [1:0]      		  		wb;
	wire [`NUM_THREADS-1:0]        	wb_valid;
	wire [`NW_BITS-1:0]       		wb_warp_num;

endinterface

`endif