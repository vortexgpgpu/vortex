`ifndef VX_GPR_JAL_IF
`define VX_GPR_JAL_IF

`include "VX_define.vh"

interface VX_gpr_jal_if ();

    wire       is_jal;
    wire[31:0] curr_PC;
    
endinterface

`endif