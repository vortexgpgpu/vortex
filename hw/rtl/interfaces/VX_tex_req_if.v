`ifndef VX_TEX_REQ_IF
`define VX_TEX_REQ_IF

`include "VX_define.vh"

interface VX_tex_req_if ();

    wire                            valid;      
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;    
    wire [`NR_BITS-1:0]             rd;    
    wire                            wb;

    wire [`NTEX_BITS-1:0]           unit;
    wire [`NUM_THREADS-1:0][31:0]   u;
    wire [`NUM_THREADS-1:0][31:0]   v;
    wire [`NUM_THREADS-1:0][31:0]   lod;
    
    wire                            ready;

endinterface
`endif


 