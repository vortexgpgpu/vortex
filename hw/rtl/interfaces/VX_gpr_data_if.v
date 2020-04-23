
`ifndef VX_GPR_DATA_IF
`define VX_GPR_DATA_IF

`include "VX_define.vh"

interface VX_gpr_data_if ();

    wire [`NUM_THREADS-1:0][31:0] a_reg_data;
    wire [`NUM_THREADS-1:0][31:0] b_reg_data;

endinterface

`endif