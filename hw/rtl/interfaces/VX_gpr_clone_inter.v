
`ifndef VX_GPR_CLONE_INTER
`define VX_GPR_CLONE_INTER

`include "../VX_define.vh"

interface VX_gpr_clone_inter ();
    /* verilator lint_off UNUSED */
    wire                is_clone;
    wire[`NW_BITS-1:0]  warp_num;
    /* verilator lint_on UNUSED */
endinterface

`endif