`ifndef VX_ALU_REQ_IF
`define VX_ALU_REQ_IF

`include "VX_define.vh"

interface VX_alu_req_if ();

    wire                    valid;   
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [31:0]             next_PC;
    wire [`ALU_BITS-1:0]    op_type;
    wire [`MOD_BITS-1:0]    op_mod;
    wire                    rs1_is_PC;
    wire                    rs2_is_imm;
    wire [31:0]             imm;
    wire [`NT_BITS-1:0]     tid;
    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;    
    wire                    ready;

endinterface

`endif