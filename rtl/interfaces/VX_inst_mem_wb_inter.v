
`include "../VX_define.v"

`ifndef VX_MEM_WB_INST_INTER

`define VX_MEM_WB_INST_INTER

interface VX_inst_mem_wb_inter ();

		wire[`NT_M1:0][31:0] alu_result;
		wire[`NT_M1:0][31:0] mem_result;
		wire[4:0]            rd;
		wire[1:0]            wb;
		wire[31:0]           PC_next;
		wire[`NT_M1:0]       valid;
		wire[`NW_M1:0]       warp_num;


endinterface


`endif