
`ifndef VX_gpr_data_INTER
`define VX_gpr_data_INTER

`include "VX_define.vh"

interface VX_gpr_data_if ();

    wire [`NUM_THREADS-1:0][31:0] a_reg_data;
    wire [`NUM_THREADS-1:0][31:0] b_reg_data;

endinterface

`endif