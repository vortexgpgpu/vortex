`ifndef VX_WSTALL_IF
`define VX_WSTALL_IF

`include "VX_define.vh"

interface VX_wstall_if();

    wire                valid;    
    wire [`NW_BITS-1:0] wid;

endinterface

`endif