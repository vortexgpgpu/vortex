
`include "VX_define.v"

`ifndef VX_MEM_WB_INST_INTER

`define VX_MEM_WB_INST_INTER

interface VX_inst_mem_wb_inter ();

		wire[`NT_M1:0][31:0] alu_result;
		wire[`NT_M1:0][31:0] mem_result;
		wire[4:0]            rd;
		wire[1:0]            wb;
		wire[4:0]            rs1;
		wire[4:0]            rs2;
		wire[31:0]           PC_next;
		wire[`NT_M1:0]       valid;
		wire[`NW_M1:0]       warp_num;

	// source-side view
	modport snk (
		input alu_result,
		input mem_result,
		input rd,
		input wb,
		input rs1,
		input rs2,
		input PC_next,
		input valid,
		input warp_num
	);


	// source-side view
	modport src (
		output alu_result,
		output mem_result,
		output rd,
		output wb,
		output rs1,
		output rs2,
		output PC_next,
		output valid,
		output warp_num
	);


endinterface


`endif