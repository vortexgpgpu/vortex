`ifndef VX_IFETCH_REQ_IF
`define VX_IFETCH_REQ_IF

`include "VX_define.vh"

interface VX_ifetch_req_if ();

    wire [`NUM_THREADS-1:0]   valid;    
    wire [`NW_BITS-1:0]       warp_num;
    wire [31:0]               curr_PC;
    wire                      ready;

endinterface

`endif