`include "VX_raster_define.vh"

interface VX_raster_req_if #(
    parameter NUM_LANES = 1
) ();

    wire                            valid;
    wire [NUM_LANES-1:0]            tmask;    
    raster_stamp_t [NUM_LANES-1:0]  stamps;
    wire                            empty;    
    wire                            ready;

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
