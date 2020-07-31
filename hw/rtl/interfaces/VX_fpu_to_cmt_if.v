`ifndef VX_FPU_TO_CMT_IF
`define VX_FPU_TO_CMT_IF

`include "VX_define.vh"

interface VX_fpu_to_cmt_if ();

    wire                            valid;
    wire [`ISTAG_BITS-1:0]          issue_tag;     
    wire [`NUM_THREADS-1:0][31:0]	data;  
    wire                            has_fflags;
    wire [`NUM_THREADS-1:0][`FFG_BITS-1:0] fflags;
    wire                            ready;

endinterface

`endif