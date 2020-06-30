`ifndef VX_WB_IF
`define VX_WB_IF

`include "VX_define.vh"

interface VX_wb_if ();

    wire [`NUM_THREADS-1:0]         valid;
    wire [`NUM_THREADS-1:0][31:0]	data; 
    wire [`NW_BITS-1:0]             warp_num;
    wire [4:0]                      rd;
    wire [1:0]                      wb;    
    wire [31:0]                     curr_PC;    

`IGNORE_WARNINGS_BEGIN
    wire                            is_io;
`IGNORE_WARNINGS_END
endinterface

`endif
