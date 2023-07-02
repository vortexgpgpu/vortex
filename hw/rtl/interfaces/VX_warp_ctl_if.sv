`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

interface VX_warp_ctl_if ();

    wire                valid;
    wire [`UP(`NW_BITS)-1:0] wid;
    gpu_tmc_t           tmc;
    gpu_wspawn_t        wspawn;
    gpu_split_t         split;
    gpu_join_t          sjoin;
    gpu_barrier_t       barrier;
    wire [`PD_STACK_SIZEW-1:0] split_ret;

    modport master (
        output valid,
        output wid,
        output wspawn,
        output tmc,
        output split,
        output sjoin,
        output barrier,
        input  split_ret
    );

    modport slave (
        input  valid,
        input  wid,
        input  wspawn,
        input  tmc,
        input  split,
        input  sjoin,
        input  barrier,
        output split_ret
    );

endinterface
