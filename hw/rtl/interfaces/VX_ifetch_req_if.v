`ifndef VX_IFETCH_REQ_IF
`define VX_IFETCH_REQ_IF

`include "VX_define.vh"

interface VX_ifetch_req_if ();

    wire                    valid;    
    wire [`NUM_THREADS-1:0] tmask;    
    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             PC;
    wire                    ready;

endinterface

`endif