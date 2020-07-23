`ifndef VX_FPU_FROM_CSR_IF
`define VX_FPU_FROM_CSR_IF

`include "VX_define.vh"

interface VX_fpu_from_csr_if ();

`IGNORE_WARNINGS_BEGIN

	wire [`NUM_WARPS-1:0][`FRM_BITS-1:0] frm;

`IGNORE_WARNINGS_END

endinterface

`endif