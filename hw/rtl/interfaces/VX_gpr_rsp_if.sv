`ifndef VX_GPR_RSP_IF
`define VX_GPR_RSP_IF

`include "VX_define.vh"

interface VX_gpr_rsp_if ();  
    
    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;

    modport master (
        output rs1_data,
        output rs2_data,
        output rs3_data
    );

    modport slave (
        input rs1_data,
        input rs2_data,
        input rs3_data
    );

endinterface

`endif
