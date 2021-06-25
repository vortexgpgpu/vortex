`ifndef VX_INSTR_SCHED_IF
`define VX_INSTR_SCHED_IF

`include "VX_define.vh"

interface VX_instr_sched_if ();

    wire                    valid;    
    wire [`NW_BITS-1:0]     wid;
    wire [`NW_BITS-1:0]     wid_n;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`EX_BITS-1:0]     ex_type;    
    wire [`OP_BITS-1:0]     op_type; 
    wire [`MOD_BITS-1:0]    op_mod;    
    wire                    wb;
    wire [`NR_BITS-1:0]     rd;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;
    wire [`NR_BITS-1:0]     rs3;
    wire [31:0]             imm; 
    wire                    use_PC;
    wire                    use_imm;
    wire [`NUM_REGS-1:0]    used_regs;      
    wire                    ready;

endinterface

`endif