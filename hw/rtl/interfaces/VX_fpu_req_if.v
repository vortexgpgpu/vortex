`ifndef VX_FPU_REQ_IF
`define VX_FPU_REQ_IF

`include "VX_define.vh"

`ifndef EXTF_F_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_fpu_req_if ();

    wire                    valid;    
    wire [`ISTAG_BITS-1:0]  issue_tag;
    wire [`NW_BITS-1:0]     wid;
`DEBUG_BEGIN
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [31:0]             curr_PC;
`DEBUG_END    
    
    wire [`FPU_BITS-1:0]    op;
    wire [`FRM_BITS-1:0]    frm;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NUM_THREADS-1:0][31:0] rs3_data;
    
    wire                    ready;

endinterface

`endif