`ifndef VX_GPU_REQ_IF
`define VX_GPU_REQ_IF

`include "VX_define.vh"

interface VX_gpu_req_if();

    wire [`NUM_THREADS-1:0]  valid;
    wire [`NW_BITS-1:0]      warp_num;
    wire [31:0]              next_PC;

    wire [`GPU_BITS-1:0]     gpu_op;

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [31:0]              rs2_data;
    
    wire                     ready;

endinterface

`endif