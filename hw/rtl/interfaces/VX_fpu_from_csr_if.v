`ifndef VX_FPU_FROM_CSR_IF
`define VX_FPU_FROM_CSR_IF

`include "VX_define.vh"

interface VX_fpu_from_csr_if ();

	wire [`NW_BITS-1:0] warp_num;
	wire [`FRM_BITS-1:0] frm;

endinterface

`endif