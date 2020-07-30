`ifndef VX_GPR_READ_IF
`define VX_GPR_READ_IF

`include "VX_define.vh"

interface VX_gpr_read_if ();  

    wire                    valid;

    wire [`NW_BITS-1:0]     warp_num;
    
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;  
    wire [`NR_BITS-1:0]     rs3;
    
    wire                    use_rs3;   

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;

    wire                    ready;

endinterface

`endif