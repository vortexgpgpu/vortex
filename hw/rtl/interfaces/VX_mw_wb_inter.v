
`include "../VX_define.vh"

`ifndef VX_MW_WB_INTER

`define VX_MW_WB_INTER

interface VX_mw_wb_inter ();

		wire[`NUM_THREADS-1:0][31:0] alu_result;
		wire[`NUM_THREADS-1:0][31:0] mem_result;
		wire[4:0]            rd;
		wire[1:0]            wb;
		wire[31:0]           PC_next;
		wire[`NUM_THREADS-1:0]       valid;
		wire [`NW_BITS-1:0]      warp_num;


endinterface


`endif