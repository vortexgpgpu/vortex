`ifndef VX_FPU_TO_CMT_IF
`define VX_FPU_TO_CMT_IF

`include "VX_define.vh"

interface VX_fpu_to_cmt_if ();

    wire                    valid;   
       
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;    
    wire [31:0]             PC;
    wire [`NUM_THREADS-1:0][31:0] data;
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    wire                    has_fflags;
    fflags_t [`NUM_THREADS-1:0] fflags;

    wire                    ready;

endinterface

`endif