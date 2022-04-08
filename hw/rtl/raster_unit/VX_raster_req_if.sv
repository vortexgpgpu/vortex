`ifndef VX_RASTER_REQ_IF
`define VX_RASTER_REQ_IF

`include "VX_raster_define.vh"

import VX_raster_types::*;

interface VX_raster_req_if ();

    wire                                valid;
    wire [`NUM_THREADS-1:0]             tmask;    
    raster_stamp_t [`NUM_THREADS-1:0]   stamps;
    wire                                empty;    
    wire                                ready;

    modport master (
        output valid,
        output tmask,
        output stamps,
        output empty,        
        input  ready
    );

    modport slave (
        input  valid,
        input  tmask,
        input  stamps,
        input  empty,
        output ready
    );

endinterface

`endif
