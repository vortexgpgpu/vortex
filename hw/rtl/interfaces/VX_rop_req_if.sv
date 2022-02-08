`ifndef VX_ROP_REQ_IF
`define VX_ROP_REQ_IF

`include "VX_define.vh"

interface VX_rop_req_if ();

    wire                            valid;

    wire [`UUID_BITS-1:0]           uuid;
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;    
    wire [`NR_BITS-1:0]             rd;    
    wire                            wb;

    wire [15:0]                     x;
    wire [15:0]                     y;
    wire [31:0]                     color;
    
    wire                            ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        output x,
        output y,
        output color,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  wb,
        input  x,
        input  y,
        input  color,
        output ready
    );

endinterface
`endif


 