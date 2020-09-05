`ifndef VX_GPR_RSP_IF
`define VX_GPR_RSP_IF

`include "VX_define.vh"

interface VX_gpr_rsp_if ();  
    wire                    valid;
`IGNORE_WARNINGS_BEGIN
    wire [`NW_BITS-1:0]     wid;
    wire [31:0]             PC;
`IGNORE_WARNINGS_END

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;

    wire                    ready;

endinterface

`endif