`ifndef VX_GPU_REQ_IF
`define VX_GPU_REQ_IF

`include "VX_define.vh"

interface VX_gpu_req_if();

    wire                    valid;
    wire [`ISTAG_BITS-1:0]  issue_tag;
    wire [`NUM_THREADS-1:0] thread_mask;
    wire [`NW_BITS-1:0]     warp_num;
`DEBUG_BEGIN
    wire [31:0]             curr_PC;
`DEBUG_END
    wire [`GPU_BITS-1:0]    gpu_op;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [31:0]             rs2_data;
    wire [31:0]             next_PC;
    
    wire                    ready;

endinterface

`endif