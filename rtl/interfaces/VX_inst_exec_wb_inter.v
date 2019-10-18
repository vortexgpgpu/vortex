
`include "../VX_define.v"

`ifndef VX_EXEC_UNIT_WB_INST_INTER

`define VX_EXEC_UNIT_WB_INST_INTER

interface VX_inst_exec_wb_inter ();

	wire[`NT_M1:0][31:0]  alu_result; 
	wire[4:0]      		  rd;
	wire[1:0]      		  wb;
	wire[`NT_M1:0]        wb_valid;
	wire[`NW_M1:0]        wb_warp_num;


endinterface


`endif