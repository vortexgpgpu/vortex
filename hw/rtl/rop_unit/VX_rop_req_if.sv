`ifndef VX_ROP_REQ_IF
`define VX_ROP_REQ_IF

`include "VX_rop_define.vh"

interface VX_rop_req_if ();

    wire                                    valid;
    wire [`NUM_THREADS-1:0]                 tmask; 
    wire [`NUM_THREADS-1:0][15:0]           pos_x;
    wire [`NUM_THREADS-1:0][15:0]           pos_y;
    wire [`NUM_THREADS-1:0][31:0]           color;
    wire [`NUM_THREADS-1:0][`ROP_DEPTH_BITS-1:0] depth;
    wire                                    backface;

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

`endif