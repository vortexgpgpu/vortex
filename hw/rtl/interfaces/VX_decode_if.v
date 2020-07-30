`ifndef VX_DECODE_IF
`define VX_DECODE_IF

`include "VX_define.vh"

interface VX_decode_if ();

    wire                    valid;    
    wire [`NW_BITS-1:0]     warp_num;
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [31:0]             curr_PC;
    wire [31:0]             next_PC;   

    wire [`EX_BITS-1:0]     ex_type;    
    wire [`OP_BITS-1:0]     ex_op; 

    wire [`NR_BITS-1:0]     rd;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;
    wire [31:0]             imm;    

    wire                    rs1_is_PC;
    wire                    rs2_is_imm;  
    
    wire [`NUM_REGS-1:0]    reg_use_mask;

    // FP states
    wire [`NR_BITS-1:0]     rs3;
    wire                    use_rs3; 
    wire [`FRM_BITS-1:0]    frm;    

    wire                    wb;

    wire                    ready;

endinterface

`endif