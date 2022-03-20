`ifndef VX_RASTER_REQ_IF
`define VX_RASTER_REQ_IF

`include "VX_raster_define.vh"

interface VX_raster_req_if ();

    wire                                valid;
    wire [`NUM_THREADS-1:0]             tmask;    
    raster_stamp_t [`NUM_THREADS-1:0]   stamp;
    wire [31:0]                         empty;    
    wire                                ready;

    modport master (
        output valid,
        output tmask,
        input  stamp,
        input  empty,        
        input  ready
    );

    modport slave (
        input  valid,
        input  tmask,
        output stamp,
        output empty,
        output ready
    );

endinterface

`endif