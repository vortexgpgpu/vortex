`ifndef VX_WB_IF
`define VX_WB_IF

`include "VX_define.vh"

interface VX_wb_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [`NW_BITS-1:0]             warp_num;
    wire [31:0]                     curr_PC;  
    wire [`NUM_THREADS-1:0][31:0]	data; 
    wire [`NR_BITS-1:0]             rd;
    wire [`WB_BITS-1:0]             wb;        
    wire                            is_io;
    wire                            ready;

endinterface

`endif
