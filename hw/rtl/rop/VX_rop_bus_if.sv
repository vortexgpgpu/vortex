`include "VX_rop_define.vh"

interface VX_rop_bus_if #(
    parameter NUM_LANES = 1
) ();

    wire                                    req_valid;
    
    wire [`UP(`UUID_BITS)-1:0]              req_uuid;
    wire [NUM_LANES-1:0]                    req_mask; 
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] req_pos_x;
    wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0] req_pos_y;
    rgba_t [NUM_LANES-1:0]                  req_color;
    wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] req_depth;
    wire [NUM_LANES-1:0]                    req_face;

    wire                                    req_ready;

    modport master (
        output req_valid,
        output req_uuid,
        output req_mask,
        output req_pos_x,
        output req_pos_y,
        output req_color,
        output req_depth,
        output req_face,
        input  req_ready
    );

    modport slave (
        input  req_valid,
        input  req_uuid,
        input  req_mask,
        input  req_pos_x,
        input  req_pos_y,
        input  req_color,
        input  req_depth,
        input  req_face,
        output req_ready
    );

endinterface
