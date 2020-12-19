`ifndef VX_MUL_REQ_IF
`define VX_MUL_REQ_IF

`include "VX_define.vh"

`ifndef EXT_M_ENABLE
    `IGNORE_WARNINGS_BEGIN
`endif

interface VX_mul_req_if ();

    wire                    valid;
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`MUL_BITS-1:0]    op_type;
    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data;
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;        
    wire                    ready;

endinterface

`endif