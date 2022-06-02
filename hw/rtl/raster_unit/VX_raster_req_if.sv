`include "VX_raster_define.vh"

interface VX_raster_req_if #(
    parameter NUM_LANES = 1
) ();

    wire                            valid;  
    raster_stamp_t [NUM_LANES-1:0]  stamps;
    wire                            empty;    
    wire                            ready;

    modport master (
        output valid,
        output stamps,
        output empty,        
        input  ready
    );

    modport slave (
        input  valid,
        input  stamps,
        input  empty,
        output ready
    );

endinterface
