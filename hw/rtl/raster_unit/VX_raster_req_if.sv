`include "VX_raster_define.vh"

interface VX_raster_req_if #(
    parameter NUM_LANES = 1
) ();

    wire                            valid;  
    raster_stamp_t [NUM_LANES-1:0]  stamps;
    wire                            done;    
    wire                            ready;

    modport master (
        output valid,
        output stamps,
        output done,        
        input  ready
    );

    modport slave (
        input  valid,
        input  stamps,
        input  done,
        output ready
    );

endinterface
