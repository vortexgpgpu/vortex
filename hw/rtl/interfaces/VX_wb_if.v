`ifndef VX_WB_IF
`define VX_WB_IF

`include "VX_define.vh"

interface VX_wb_if ();

    wire                            valid;
    wire [`NUM_THREADS-1:0]         thread_mask;
    wire [`NW_BITS-1:0]             wid; 

`IGNORE_WARNINGS_BEGIN
    wire [31:0]                     curr_PC;
`IGNORE_WARNINGS_END

    wire [`NR_BITS-1:0]             rd;
    wire [`NUM_THREADS-1:0][31:0]	data; 

endinterface

`endif
