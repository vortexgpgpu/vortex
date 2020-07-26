`ifndef VX_FPU_TO_CSR_IF
`define VX_FPU_TO_CSR_IF

`include "VX_define.vh"

`ifndef EXTF_F_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_fpu_to_csr_if ();

	wire valid;

	wire [`NW_BITS-1:0] warp_num;

	wire fflags_NV;
	wire fflags_DZ;
	wire fflags_OF;
	wire fflags_UF;
	wire fflags_NX;

endinterface

`endif