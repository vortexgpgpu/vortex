`ifndef VX_MUL_REQ_IF
`define VX_MUL_REQ_IF

`include "VX_define.vh"

`ifndef EXT_M_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_mul_req_if ();

    wire                    valid;
    wire [`ISTAG_BITS-1:0]  issue_tag;
    
    wire [`MUL_BITS-1:0]    mul_op;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
        
    wire                    ready;

endinterface

`endif