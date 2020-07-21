`ifndef VX_MUL_REQ_IF
`define VX_MUL_REQ_IF

`include "VX_define.vh"

interface VX_mul_req_if ();

    wire [`NUM_THREADS-1:0]     valid;
    wire [`NW_BITS-1:0]         warp_num;
    wire [31:0]                 curr_PC;
    
    wire [`MUL_BITS-1:0]        mul_op;

    wire [`WB_BITS-1:0]         wb;
    wire [`NR_BITS-1:0]         rd;    

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
        
    wire                        ready;

endinterface

`endif