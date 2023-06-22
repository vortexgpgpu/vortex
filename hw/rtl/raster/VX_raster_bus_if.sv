`include "VX_raster_define.vh"

interface VX_raster_bus_if #(
    parameter NUM_LANES = 1
) ();

    wire                            req_valid;  
    raster_stamp_t [NUM_LANES-1:0]  req_stamps;
    wire                            req_done;    
    wire                            req_ready;

    modport master (
        output req_valid,
        output req_stamps,
        output req_done,        
        input  req_ready
    );

    modport slave (
        input  req_valid,
        input  req_stamps,
        input  req_done,
        output req_ready
    );

endinterface
