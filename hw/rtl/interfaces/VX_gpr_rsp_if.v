`ifndef VX_GPR_RSP_IF
`define VX_GPR_RSP_IF

`include "VX_define.vh"

interface VX_gpr_rsp_if ();  
    
    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;

endinterface

`endif