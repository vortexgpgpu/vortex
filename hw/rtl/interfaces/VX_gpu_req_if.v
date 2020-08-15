`ifndef VX_GPU_REQ_IF
`define VX_GPU_REQ_IF

`include "VX_define.vh"

interface VX_gpu_req_if();

    wire                    valid;
    wire [`ISTAG_BITS-1:0]  issue_tag;    
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [31:0]             curr_PC;

    wire [`GPU_BITS-1:0]    op;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [31:0]             rs2_data;
    
    wire                    ready;

endinterface

`endif