`ifndef VX_ALU_REQ_IF
`define VX_ALU_REQ_IF

`include "VX_define.vh"

interface VX_alu_req_if ();

    wire                    valid;    
    wire [`ISTAG_BITS-1:0]  issue_tag;
`DEBUG_BEGIN
    wire [`NW_BITS-1:0]     wid;    
    wire [`NUM_THREADS-1:0] thread_mask;
`DEBUG_END        
    wire [31:0]             curr_PC;

    wire [`ALU_BITS-1:0]    op;

    wire                    rs1_is_PC;
    wire                    rs2_is_imm;

    wire [31:0]             imm;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    
    wire                    ready;

endinterface

`endif