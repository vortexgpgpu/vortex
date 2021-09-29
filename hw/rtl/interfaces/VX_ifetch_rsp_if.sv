`ifndef VX_IFETCH_RSP_IF
`define VX_IFETCH_RSP_IF

`include "VX_define.vh"

interface VX_ifetch_rsp_if ();

    wire                    valid;
    wire [`NUM_THREADS-1:0] tmask;    
    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             PC;
    wire [31:0]             data;
    wire                    ready;

    modport master (
        output valid,    
        output tmask,
        output wid,
        output PC,
        output data,
        input  ready
    );

    modport slave (
        input  valid,   
        input  tmask,
        input  wid,
        input  PC,
        input  data,
        output ready
    );

endinterface

`endif