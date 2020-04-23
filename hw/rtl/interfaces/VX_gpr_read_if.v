`ifndef VX_GPR_READ_IF
`define VX_GPR_READ_IF

`include "VX_define.vh"

interface VX_gpr_read_if ();

    wire [4:0]          rs1;
    wire [4:0]          rs2;
    wire [`NW_BITS-1:0] warp_num;

endinterface

`endif