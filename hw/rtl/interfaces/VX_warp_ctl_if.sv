`include "VX_define.vh"

import VX_gpu_types::*;

interface VX_warp_ctl_if ();

    wire                valid;
    wire [`NW_BITS-1:0] wid;
    gpu_tmc_t           tmc;
    gpu_wspawn_t        wspawn;
    gpu_barrier_t       barrier;
    gpu_split_t         split;

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
