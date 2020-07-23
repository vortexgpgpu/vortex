`ifndef VX_FPU_TO_CSR_IF
`define VX_FPU_TO_CSR_IF

`include "VX_define.vh"

interface VX_fpu_to_csr_if ();

`IGNORE_WARNINGS_BEGIN
	wire valid;

	wire [`NW_BITS-1:0] warp_num;

	wire fflags_NV;
	wire fflags_DZ;
	wire fflags_OF;
	wire fflags_UF;
	wire fflags_NX;

`IGNORE_WARNINGS_END

endinterface

`endif