`include "VX_rop_define.vh"

interface VX_rop_req_if #(
    parameter NUM_LANES = 1
) ();

    wire                                    valid;
    
    wire [`UP(`UUID_BITS)-1:0]              uuid;
    wire [NUM_LANES-1:0]                    mask; 
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] pos_x;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] pos_y;
    rgba_t [NUM_LANES-1:0]                  color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] depth;
    wire [NUM_LANES-1:0]                    face;

    wire                                    ready;

    modport master (
        output valid,
        output uuid,
        output mask,
        output pos_x,
        output pos_y,
        output color,
        output depth,
        output face,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  mask,
        input  pos_x,
        input  pos_y,
        input  color,
        input  depth,
        input  face,
        output ready
    );

endinterface
