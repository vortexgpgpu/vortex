`include "VX_rop_define.vh"

interface VX_rop_req_if #(
    parameter NUM_LANES = 1
) ();

    wire                                    valid;
    
    wire [NUM_LANES-1:0]                    tmask; 
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] pos_x;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] pos_y;
    rgba_t [NUM_LANES-1:0]                  color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] depth;
    wire [NUM_LANES-1:0]                    backface;

    wire                                    ready;

    modport master (
        output valid,
        output tmask,
        output pos_x,
        output pos_y,
        output color,
        output depth,
        output backface,
        input  ready
    );

    modport slave (
        input  valid,
        input  tmask,
        input  pos_x,
        input  pos_y,
        input  color,
        input  depth,
        input  backface,
        output ready
    );

endinterface
