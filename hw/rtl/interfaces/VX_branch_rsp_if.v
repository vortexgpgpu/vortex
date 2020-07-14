`ifndef VX_BRANCH_RSP_IF
`define VX_BRANCH_RSP_IF

`include "VX_define.vh"

interface VX_branch_rsp_if ();

    wire                valid;
    wire                dir;
    wire [31:0]         dest;
    wire [`NW_BITS-1:0] warp_num;

endinterface

`endif