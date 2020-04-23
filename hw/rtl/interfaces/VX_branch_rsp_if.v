`ifndef VX_BRANCH_RSP_IF
`define VX_BRANCH_RSP_IF

`include "VX_define.vh"

interface VX_branch_rsp_if ();

    wire                valid_branch;
    wire                branch_dir;
    wire [31:0]         branch_dest;
    wire [`NW_BITS-1:0] branch_warp_num;

endinterface

`endif