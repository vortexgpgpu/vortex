
`include "../VX_define.v"

`ifndef VX_CSR_WB_REQ

`define VX_CSR_WB_REQ

interface VX_csr_wb_inter ();

	wire[`NT_M1:0]        valid;
	wire[`NW_M1:0]        warp_num;
	wire[4:0]             rd;
	wire[1:0]             wb;
	
	wire[`NT_M1:0][31:0]  csr_result;


endinterface


`endif