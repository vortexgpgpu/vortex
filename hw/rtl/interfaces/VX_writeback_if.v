`ifndef VX_WRITEBACK_IF
`define VX_WRITEBACK_IF

`include "VX_define.vh"

interface VX_writeback_if ();

    wire                            valid;

    wire [`NUM_THREADS-1:0]         tmask;
    wire [`NW_BITS-1:0]             wid; 
    wire [31:0]                     PC;
    wire [`NR_BITS-1:0]             rd;
    wire [`NUM_THREADS-1:0][31:0]   data;
    wire                            eop;
    
    wire                            ready;

endinterface

`endif
