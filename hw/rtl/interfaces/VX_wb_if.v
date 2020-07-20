`ifndef VX_WB_IF
`define VX_WB_IF

`include "VX_define.vh"

interface VX_wb_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [`NW_BITS-1:0]             warp_num; 
    wire [`NUM_THREADS-1:0][31:0]	data; 
    wire [`NR_BITS-1:0]             rd;
    wire                            ready;

endinterface

`endif
