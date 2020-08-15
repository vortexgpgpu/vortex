`ifndef VX_BRANCH_RSP_IF
`define VX_BRANCH_RSP_IF

`include "VX_define.vh"

interface VX_branch_ctl_if ();

    wire                valid;    
    wire [`NW_BITS-1:0] wid;    
    wire                taken;
    wire [31:0]         dest;

endinterface

`endif