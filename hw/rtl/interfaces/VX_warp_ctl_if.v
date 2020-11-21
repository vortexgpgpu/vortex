`ifndef VX_WARP_CTL_IF
`define VX_WARP_CTL_IF

`include "VX_define.vh"

interface VX_warp_ctl_if ();

    wire                valid;
    wire [`NW_BITS-1:0] wid;
    gpu_tmc_t           tmc;
    gpu_wspawn_t        wspawn;
    gpu_barrier_t       barrier;
    gpu_split_t         split;

endinterface

`endif