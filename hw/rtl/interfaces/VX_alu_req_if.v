`ifndef VX_ALU_REQ_IF
`define VX_ALU_REQ_IF

`include "VX_define.vh"

interface VX_alu_req_if ();

    wire [`NUM_THREADS-1:0] valid;    
    wire [`NW_BITS-1:0]     warp_num;
    wire [31:0]             curr_PC;
    
    wire [`ALU_BITS-1:0]    alu_op;

    wire [`WB_BITS-1:0]     wb;
    wire [`NR_BITS-1:0]     rd;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;

    wire [31:0]             offset;
    wire [31:0]             next_PC;    
    
    wire                    ready;

endinterface

`endif