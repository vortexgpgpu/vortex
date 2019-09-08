
`include "VX_define.v"

`ifndef VX_FWD_CSR_RSP

`define VX_FWD_CSR_RSP

interface VX_forward_csr_response_inter ();
		/* verilator lint_off UNUSED */
		wire                 csr_fwd;
		wire[31:0]           csr_fwd_data;
		/* verilator lint_on UNUSED */

	// source-side view
	modport snk (

		input csr_fwd,
		input csr_fwd_data
	);


	// source-side view
	modport src (
		output csr_fwd,
		output csr_fwd_data
	);


endinterface


`endif