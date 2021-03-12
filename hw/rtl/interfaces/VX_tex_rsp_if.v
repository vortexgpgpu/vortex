`ifndef VX_TEX_RSP_IF
`define VX_TEX_RSP_IF

`include "VX_define.vh"

interface VX_tex_rsp_if ();
    // wire                 valid;   
    // wire [`TAGW-1:0]     tag;
    wire [`NUM_THREADS-1:0][31:0]  data;
    wire                 ready;
endinterface
`endif
 
 
