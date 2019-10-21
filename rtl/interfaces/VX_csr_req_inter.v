
`include "../VX_define.v"

`ifndef VX_CSR_REQ

`define VX_CSR_REQ

interface VX_csr_req_inter ();

	wire[`NT_M1:0] valid;
	wire[`NW_M1:0] warp_num;
	wire[4:0]      rd;
	wire[1:0]      wb;

	wire           is_csr;
	wire[11:0]     csr_address;
	wire           csr_immed;
	wire[31:0]     csr_mask;


endinterface


`endif