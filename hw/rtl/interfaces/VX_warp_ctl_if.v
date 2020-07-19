`ifndef VX_WARP_CTL_IF
`define VX_WARP_CTL_IF

`include "VX_define.vh"

interface VX_warp_ctl_if ();

    wire [`NW_BITS-1:0]     warp_num;

    wire                    change_mask;
    wire [`NUM_THREADS-1:0] thread_mask;

    wire                    wspawn;
    wire [31:0]             wspawn_pc;
    wire [`NUM_WARPS-1:0]   wspawn_new_active;

    wire                    whalt;

    wire                    is_barrier;
    wire [`NB_BITS-1:0]     barrier_id;
    wire [`NW_BITS:0]       num_warps;

    wire                    is_split;
    wire                 	do_split;

    wire [`NUM_THREADS-1:0] split_new_mask;
    wire [`NUM_THREADS-1:0] split_later_mask;
    wire [31:0]             split_save_pc;

endinterface

`endif