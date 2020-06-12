`ifndef VX_GPGPU_INST_REQ_IF
`define VX_GPGPU_INST_REQ_IF

`include "VX_define.vh"

interface VX_gpu_inst_req_if();

    wire [`NUM_THREADS-1:0]  valid;
    wire [`NW_BITS-1:0]      warp_num;
    wire                     is_wspawn;
    wire                     is_tmc;   
    wire                     is_split; 

    wire                     is_barrier;

    wire[31:0]               next_PC;

    wire [`NUM_THREADS-1:0][31:0] a_reg_data;
    wire [31:0]              rd2;

endinterface

`endif