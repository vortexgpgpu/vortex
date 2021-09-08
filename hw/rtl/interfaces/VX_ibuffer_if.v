`ifndef VX_IBUFFER_IF
`define VX_IBUFFER_IF

`include "VX_define.vh"

interface VX_ibuffer_if ();

    wire                    valid;    
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`EX_BITS-1:0]     ex_type;    
    wire [`INST_OP_BITS-1:0] op_type; 
    wire [`INST_MOD_BITS-1:0] op_mod;    
    wire                    wb;
    wire                    use_PC;
    wire                    use_imm;
    wire [31:0]             imm;
    wire [`NR_BITS-1:0]     rd;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;
    wire [`NR_BITS-1:0]     rs3;
    wire                    ready;

    // scoreboard forwarding
    wire [`NR_BITS-1:0]     rd_n;
    wire [`NR_BITS-1:0]     rs1_n;
    wire [`NR_BITS-1:0]     rs2_n;
    wire [`NR_BITS-1:0]     rs3_n;
    wire [`NW_BITS-1:0]     wid_n;
    
endinterface

`endif