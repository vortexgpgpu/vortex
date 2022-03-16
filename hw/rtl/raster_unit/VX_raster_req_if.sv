`ifndef VX_RASTER_REQ_IF
`define VX_RASTER_REQ_IF

`include "VX_raster_define.vh"

interface VX_raster_req_if ();

    wire                                valid;
    wire [31:0]                         empty;
    raster_stamp_t [`NUM_THREADS-1:0]   stamp;    
    wire                                ready;

    modport master (
        output valid,
        input  empty,
        input  stamp,
        input  ready
    );

    modport slave (
        input  valid,
        output empty,
        output stamp,
        output ready
    );

endinterface

`endif