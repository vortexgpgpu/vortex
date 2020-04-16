
`include "../VX_define.vh"

`ifndef VX_LSU_REQ_INTER

`define VX_LSU_REQ_INTER

interface VX_lsu_req_inter ();

	wire[`NUM_THREADS-1:0]       valid;
	wire[31:0]           lsu_pc;
	wire[`NW_BITS-1:0]       warp_num;
	wire[`NUM_THREADS-1:0][31:0] store_data;
	wire[`NUM_THREADS-1:0][31:0] base_address; // A reg data
	wire[31:0]           offset;       // itype_immed
	wire[2:0]            mem_read; 
	wire[2:0]            mem_write;
	wire[4:0]            rd;
	wire[1:0]            wb;

endinterface


`endif