`ifndef VX_RASTER_RSP_IF
`define VX_RASTER_RSP_IF

`include "VX_raster_define.vh"

interface VX_raster_rsp_if ();

    wire                    valid;

    wire [`UUID_BITS-1:0]   uuid;
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;    
    wire [31:0]             PC;    
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    
    wire [31:0]             rem;

    wire                    ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        output rem,
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
        input  rem,
        output ready
    );

endinterface

`endif
 
 
