
`ifndef VX_JAL_RSP_IF
`define VX_JAL_RSP_IF

`include "VX_define.vh"

interface VX_jal_rsp_if ();

    wire                valid;
    wire [31:0]         dest;
    wire [`NW_BITS-1:0] warp_num;
    
endinterface

`endif