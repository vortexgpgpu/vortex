`ifndef VX_TEX_REQ_IF
`define VX_TEX_REQ_IF

`include "VX_define.vh"

interface VX_tex_req_if ();
    wire                                valid;       
    wire [`NUM_THREADS-1:0][31:0]           u;
    wire [`NUM_THREADS-1:0][31:0]           v;
    wire [`NUM_THREADS-1:0][31:0]           lod_t;
    // wire [`MADDRW-1:0]                   addr;    
    // wire [`MAXWTW-1:0]                   width;
    // wire [`MAXHTW-1:0]                   height;
    // wire [`MAXFTW-1:0]                   format;
    // wire [`MAXFMW-1:0]                   filter;
    // wire [`MAXAMW-1:0]                   clamp;
    // wire [`TAGW-1:0]                     tag;
    // wire                                ready; 

endinterface
`endif


 