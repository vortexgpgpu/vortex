`ifndef VX_RASTER_TO_ROP_IF
`define VX_RASTER_TO_ROP_IF

`include "VX_raster_define.vh"

interface VX_raster_to_rop_if ();

    wire                            valid;
    wire [`NW_BITS-1:0]             wid;
    
    logic [`NUM_THREADS-1:0][15:0]  pos_x;
    logic [`NUM_THREADS-1:0][15:0]  pos_y;
    logic [`NUM_THREADS-1:0][3:0]   mask;

    wire                            ready;

    modport master (
        output valid,
        output wid,
        input  pos_x,
        input  pos_y,
        input  mask,
        input  ready
    );

    modport slave (
        input   valid,
        input   wid,
        output  pos_x,
        output  pos_y,
        output  mask,
        output  ready
    );

endinterface

`endif
 
 
