`ifndef VX_GPR_REQ_IF
`define VX_GPR_REQ_IF

`include "VX_define.vh"

interface VX_gpr_req_if ();  
  
    wire [`NW_BITS-1:0] wid;
    wire [`NR_BITS-1:0] rs1;
    wire [`NR_BITS-1:0] rs2;  
    wire [`NR_BITS-1:0] rs3; 

endinterface

`endif