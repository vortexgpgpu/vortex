`ifndef VX_CMT_TO_CSR_IF
`define VX_CMT_TO_CSR_IF

`include "VX_define.vh"

interface VX_cmt_to_csr_if ();

    wire valid;
    wire [`NE_BITS:0] num_commits;

    wire upd_fflags;
    wire [`NW_BITS-1:0] fpu_warp_num;
    wire fflags_NV;
	wire fflags_DZ;
	wire fflags_OF;
	wire fflags_UF;
	wire fflags_NX;   

endinterface

`endif