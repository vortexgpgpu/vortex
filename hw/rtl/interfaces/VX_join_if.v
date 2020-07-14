
`ifndef VX_JOIN_IF
`define VX_JOIN_IF

`include "VX_define.vh"

interface VX_join_if ();

    wire                is_join;
    wire [`NW_BITS-1:0] warp_num;

endinterface

`endif