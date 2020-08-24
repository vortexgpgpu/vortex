`ifndef VX_ISSUE_IF
`define VX_ISSUE_IF

`include "VX_define.vh"

interface VX_issue_if ();

    wire                    valid;    

    wire [`ITAG_BITS-1:0]  issue_tag;
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [31:0]             curr_PC;

    wire [`EX_BITS-1:0]     ex_type;    
    wire [`OP_BITS-1:0]     op_type; 

    wire [`FRM_BITS-1:0]    frm;

    wire                    wb;

    wire [`NR_BITS-1:0]     rd;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;
    
    wire [`NR_BITS-1:0]     rs1;
    wire [31:0]             imm;

    wire                    rs1_is_PC;
    wire                    rs2_is_imm;

    wire [1NT_BITS-1:0]     tid;

endinterface

`endif