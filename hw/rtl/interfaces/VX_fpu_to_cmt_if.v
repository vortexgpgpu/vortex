`ifndef VX_FPU_TO_CMT_IF
`define VX_FPU_TO_CMT_IF

`include "VX_define.vh"

interface VX_fpu_to_cmt_if ();

    wire                            valid;
    wire [`ISTAG_BITS-1:0]          issue_tag;     
    wire [`NUM_THREADS-1:0][31:0]	data;  
    wire                            upd_fflags;
    wire                            fflags_NV;
	wire                            fflags_DZ;
	wire                            fflags_OF;
	wire                            fflags_UF;
	wire                            fflags_NX;
    wire                            ready;

endinterface

`endif