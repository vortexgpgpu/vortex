
`include "../VX_define.v"

`ifndef VX_CSR_W_REQ

`define VX_CSR_W_REQ

interface VX_csr_write_request_inter ();

		wire        is_csr;
		wire[11:0]  csr_address;
		wire[31:0]  csr_result;


endinterface


`endif