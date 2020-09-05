`ifndef VX_GPR_REQ_IF
`define VX_GPR_REQ_IF

`include "VX_define.vh"

interface VX_gpr_req_if ();  

    wire                    valid;

    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             PC;    
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;  
    wire [`NR_BITS-1:0]     rs3;    
    wire                    use_rs3;   

    wire                    ready;

endinterface

`endif