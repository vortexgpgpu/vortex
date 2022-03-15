`ifndef VX_ROP_SRV_IF
`define VX_ROP_SRV_IF

`include "VX_rop_define.vh"

interface VX_rop_srv_if ();

    wire                            valid;

    wire [`UUID_BITS-1:0]           uuid;
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;

    wire [`NUM_THREADS-1:0][15:0]   pos_x;
    wire [`NUM_THREADS-1:0][15:0]   pos_y;
    wire [`NUM_THREADS-1:0][31:0]   color;
    wire [`NUM_THREADS-1:0][31:0]   depth;
    
    wire                            ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output pos_x,
        output pos_y,        
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
        input  color,
        input  depth,
        output ready
    );

endinterface
`endif
 