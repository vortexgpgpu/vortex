
`include "../VX_define.v"

`ifndef VX_MEM_WB_INST_INTER

`define VX_MEM_WB_INST_INTER

interface VX_inst_mem_wb_inter ();

	wire[`NT_M1:0][31:0]  loaded_data; 
	wire[31:0]            mem_wb_pc;
	wire[4:0]      		  rd;
	wire[1:0]      		  wb;
	wire[`NT_M1:0]        wb_valid;
	wire[`NW_M1:0]        wb_warp_num;


endinterface


`endif