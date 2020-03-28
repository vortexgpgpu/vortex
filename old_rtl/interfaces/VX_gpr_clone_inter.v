
`include "../VX_define.v"

`ifndef VX_GPR_CLONE_INTER

`define VX_GPR_CLONE_INTER


interface VX_gpr_clone_inter ();
/* verilator lint_off UNUSED */
wire           is_clone;
wire[`NW_M1:0] warp_num;
/* verilator lint_on UNUSED */
endinterface



`endif