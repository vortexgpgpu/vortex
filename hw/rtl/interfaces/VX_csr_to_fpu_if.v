`ifndef VX_CSR_TO_FPU_IF
`define VX_CSR_TO_FPU_IF

`include "VX_define.vh"

`ifndef EXTF_F_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_csr_to_fpu_if ();

	wire [`NW_BITS-1:0]  warp_num;
	wire [`FRM_BITS-1:0] frm;

endinterface

`endif