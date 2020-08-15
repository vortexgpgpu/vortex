`ifndef VX_BRANCH_REQ_IF
`define VX_BRANCH_REQ_IF

`include "VX_define.vh"

interface VX_bru_req_if ();

    wire                    valid;    
    wire [`ISTAG_BITS-1:0]  issue_tag;    
    wire [`NW_BITS-1:0]     wid;
`DEBUG_BEGIN
    wire [`NUM_THREADS-1:0] thread_mask;
`DEBUG_END
    wire [31:0]             curr_PC;

    wire [`BRU_BITS-1:0]    op;
    
    wire                    rs1_is_PC;

    wire [31:0]             rs1_data;
    wire [31:0]             rs2_data;
    
    wire [31:0]             offset;
    
    wire                    ready;

endinterface

`endif