`ifndef VX_COMMIT_IF
`define VX_COMMIT_IF

`include "VX_define.vh"

interface VX_commit_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [`NW_BITS-1:0]             warp_num;
    wire [31:0]                     curr_PC;  
    wire [`NUM_THREADS-1:0][31:0]	data; 
    wire [`NR_BITS-1:0]             rd;
    wire                            wb;        
    wire                            ready;

endinterface

`endif