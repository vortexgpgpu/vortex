`ifndef VX_WARP_CTL_IF
`define VX_WARP_CTL_IF

`include "VX_define.vh"

interface VX_warp_ctl_if ();

    wire                    valid;
    wire [`NW_BITS-1:0]     wid;
    gpu_types::gpu_tmc_t    tmc;
    gpu_types::gpu_wspawn_t wspawn;
    gpu_types::gpu_barrier_t barrier;
    gpu_types::gpu_split_t  split;

    modport master (
        output valid,
        output wid,
        output tmc,
        output wspawn,
        output barrier,
        output split
    );

    modport slave (
        input valid,
        input wid,
        input tmc,
        input wspawn,
        input barrier,
        input split
    );

endinterface

`endif
