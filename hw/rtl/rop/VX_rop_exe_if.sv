`include "VX_rop_define.vh"

interface VX_rop_exe_if ();

    wire                                        valid;

    wire [`UP(`UUID_BITS)-1:0]                  uuid;
    wire [`UP(`NW_BITS)-1:0]                    wid;
    wire [`NUM_THREADS-1:0]                     tmask;    
    wire [`XLEN-1:0]                            PC;

    wire [`NUM_THREADS-1:0][`VX_ROP_DIM_BITS-1:0] pos_x;
    wire [`NUM_THREADS-1:0][`VX_ROP_DIM_BITS-1:0] pos_y;
    wire [`NUM_THREADS-1:0]                     face;
    wire [`NUM_THREADS-1:0][31:0]               color;
    wire [`NUM_THREADS-1:0][`VX_ROP_DEPTH_BITS-1:0] depth;
    
    wire                                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,   
        output pos_x,
        output pos_y,
        output face,
        output color,
        output depth,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,      
        input  pos_x,
        input  pos_y,
        input  face,
        input  color,
        input  depth,
        output ready
    );

endinterface
