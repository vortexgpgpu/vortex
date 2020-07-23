`ifndef VX_FPU_REQ_IF
`define VX_FPU_REQ_IF

`include "VX_define.vh"

interface VX_fpu_req_if ();

    wire [`NUM_THREADS-1:0] valid;    
    wire [`NW_BITS-1:0]     warp_num;
    wire [31:0]             curr_PC;
    
    wire [`FPU_BITS-1:0]    fpu_op;
    wire [`FRM_BITS-1:0]    frm;

    wire                    wb;
    wire [`NR_BITS-1:0]     rd;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;
    
    wire                    ready;

endinterface

`endif