`ifndef VX_CSR_WB_REQ
`define VX_CSR_WB_REQ

`include "../VX_define.v"

interface VX_csr_wb_if ();

	wire [`NUM_THREADS-1:0]			valid;
	wire [`NW_BITS-1:0]      		warp_num;
	wire [4:0]             			rd;
	wire [1:0]                		wb;
	
	wire [`NUM_THREADS-1:0][31:0]  	csr_result;

endinterface

`endif