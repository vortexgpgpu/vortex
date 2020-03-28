`include "../VX_define.v"

`ifndef VX_MEM_REQ_IN

`define VX_MEM_REQ_IN

interface VX_mem_req_inter ();

	wire[`NT_M1:0][31:0] alu_result;
	wire[2:0]            mem_read; 
	wire[2:0]            mem_write;
	wire[4:0]            rd;
	wire[1:0]            wb;
	wire[4:0]            rs1;
	wire[4:0]            rs2;
	wire[`NT_M1:0][31:0] rd2;
	wire[31:0]           PC_next;
	wire[31:0]           curr_PC;
	wire[31:0]           branch_offset;
	wire[2:0]            branch_type; 
	wire[`NT_M1:0]       valid;
	wire[`NW_M1:0]       warp_num;


endinterface


`endif