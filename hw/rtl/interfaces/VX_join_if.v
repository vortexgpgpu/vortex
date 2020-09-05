`ifndef VX_JOIN_IF
`define VX_JOIN_IF

`include "VX_define.vh"

interface VX_join_if ();

    wire                valid;
    wire [`NW_BITS-1:0] wid;

endinterface

`endif