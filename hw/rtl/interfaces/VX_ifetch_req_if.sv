`ifndef VX_IFETCH_REQ_IF
`define VX_IFETCH_REQ_IF

`include "VX_define.vh"

interface VX_ifetch_req_if ();

    wire                    valid;    
    wire [`NUM_THREADS-1:0] tmask;    
    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             PC;
    wire                    ready;

    modport master (
        output valid,    
        output tmask,
        output wid,
        output PC,
        input  ready
    );

    modport slave (
        input  valid,   
        input  tmask,
        input  wid,
        input  PC,
        output ready
    );

endinterface

`endif