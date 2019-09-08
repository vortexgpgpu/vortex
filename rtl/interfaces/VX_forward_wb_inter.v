
`include "VX_define.v"

`ifndef VX_FWD_WB

`define VX_FWD_WB

interface VX_forward_wb_inter ();

		wire[4:0]            dest;
		wire[1:0]            wb;
		wire[`NT_M1:0][31:0] alu_result;
		wire[`NT_M1:0][31:0] mem_data;
		wire[31:0]           PC_next;
		wire[`NW_M1:0]       warp_num;

	// source-side view
	modport snk (
		input dest,
		input wb,
		input alu_result,
		input mem_data,
		input PC_next,
		input warp_num
	);


	// source-side view
	modport src (
		output dest,
		output wb,
		output alu_result,
		output mem_data,
		output PC_next,
		output warp_num
	);


endinterface


`endif