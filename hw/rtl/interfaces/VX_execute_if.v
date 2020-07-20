`ifndef VX_EXECUTE_IF
`define VX_EXECUTE_IF

`include "VX_define.vh"

interface VX_execute_if ();

    wire [`NUM_THREADS-1:0] valid;
    wire [`NW_BITS-1:0]     warp_num;
    wire [31:0]             curr_PC;
    wire [`EX_BITS-1:0]     ex_type;    
    wire [`OP_BITS-1:0]     instr_op; 

    wire [`NR_BITS-1:0]     rd;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;
    wire [31:0]             imm;    
    wire                    rs1_is_PC;
    wire                    rs2_is_imm; 
    wire [31:0]             next_PC;     

    wire [`WB_BITS-1:0]     wb;

    wire                    alu_ready;  
    wire                    mul_ready;
    wire                    lsu_ready;  
    wire                    csr_ready;
    wire                    gpu_ready;

endinterface

`endif