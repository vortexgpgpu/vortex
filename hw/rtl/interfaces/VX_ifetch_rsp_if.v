`ifndef VX_IFETCH_RSP_IF
`define VX_IFETCH_RSP_IF

`include "VX_define.vh"

interface VX_ifetch_rsp_if ();

    wire                    valid;    
    wire [`NUM_THREADS-1:0] thread_mask;    
    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             curr_PC;
    wire [31:0]             instr;
    wire                    ready;

endinterface

`endif