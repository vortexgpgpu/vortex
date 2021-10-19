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
    wire [1:0][`NUM_THREADS-1:0][31:0] coords;
    wire [`NUM_THREADS-1:0][31:0]   lod;
    
    wire                            ready;

    modport master (
        output valid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        output unit,
        output coords,
        output lod,
        input  ready
    );

    modport slave (
        input  valid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  wb,
        input  unit,
        input  coords,
        input  lod,
        output ready
    );

endinterface
`endif


 