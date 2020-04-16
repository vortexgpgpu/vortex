`ifndef VX_CSR_REQ
`define VX_CSR_REQ

`include "../VX_define.vh"

interface VX_csr_req_inter ();

	wire[`NUM_THREADS-1:0] valid;
	wire[`NW_BITS-1:0] warp_num;
	wire[4:0]      rd;
	wire[1:0]      wb;
	wire[4:0]      alu_op;
	wire           is_csr;
	wire[11:0]     csr_address;
	wire           csr_immed;
	wire[31:0]     csr_mask;

endinterface

`endif