`include "VX_tex_define.vh"

import VX_tex_types::*;

interface VX_tex_req_if ();

    wire                            valid;      
    wire [`UUID_BITS-1:0]           uuid;
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;    
    wire [`NR_BITS-1:0]             rd;    
    wire                            wb;

    wire [1:0][`NUM_THREADS-1:0][31:0] coords;
    wire [`NUM_THREADS-1:0][31:0]   lod;
    
    wire                            ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        output coords,
        output lod,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  wb,
        input  coords,
        input  lod,
        output ready
    );

endinterface
