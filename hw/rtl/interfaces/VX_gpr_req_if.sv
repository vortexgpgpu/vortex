`ifndef VX_GPR_REQ_IF
`define VX_GPR_REQ_IF

`include "VX_define.vh"

interface VX_gpr_req_if ();  
  
    wire [`NW_BITS-1:0] wid;
    wire [`NR_BITS-1:0] rs1;
    wire [`NR_BITS-1:0] rs2;  
    wire [`NR_BITS-1:0] rs3;

    modport master (
        output wid,    
        output rs1,
        output rs2,
        output rs3
    );

    modport slave (
        input wid,    
        input rs1,
        input rs2,
        input rs3
    );

endinterface

`endif
