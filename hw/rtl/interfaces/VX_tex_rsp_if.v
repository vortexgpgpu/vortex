`ifndef VX_TEX_RSP_IF
`define VX_TEX_RSP_IF

`include "VX_define.vh"

interface VX_tex_rsp_if ();

    wire                    valid;   
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;    
    wire [31:0]             PC;    
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    wire [`NUM_THREADS-1:0][31:0] data;
    wire                    ready;

endinterface

`endif
 
 
