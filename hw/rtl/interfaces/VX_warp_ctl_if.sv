// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

interface VX_warp_ctl_if import VX_gpu_pkg::*; ();

    wire tmc_valid;
    wire split_valid;
    wire sjoin_valid;
    wire wsync_valid;
    wire [NW_WIDTH-1:0] wid;
    tmc_t     tmc;
    split_t   split;
    join_t    sjoin;

    // Pipeline drain status of the scheduler's warps, by SFU-block-local warp id
    wire [SFU_BLOCK_WARPS-1:0] warp_pending_alm_empty;

    // LSU memory-scheduler drained signal (AND-reduced across all LSU blocks).
    // Used by VX_wctl_unit to gate BAR (both vx_barrier and vx_barrier_arrive)
    // so the barrier acts as a SMEM/GMEM fence — pending LSU memory ops finish
    // before the warp suspends or the barrier arrival is registered. This is
    // the CUDA __syncthreads / OpenCL barrier(CLK_LOCAL_MEM_FENCE) contract.
    wire lsu_sched_drained;

    // IPDOM stack address and mscratch of the warp at dvstack_wid
    wire [NW_WIDTH-1:0] dvstack_wid;
    wire [DV_STACK_SIZEW-1:0] dvstack_ptr;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mscratch;

    modport master (
        output tmc_valid,
        output split_valid,
        output sjoin_valid,
        output wsync_valid,
        output wid,
        output tmc,
        output split,
        output sjoin,
        input  warp_pending_alm_empty,
        input  lsu_sched_drained,
        output dvstack_wid,
        input  dvstack_ptr,
        input  mscratch
    );

    modport slave (
        input  tmc_valid,
        input  split_valid,
        input  sjoin_valid,
        input  wsync_valid,
        input  wid,
        input  tmc,
        input  split,
        input  sjoin,
        output warp_pending_alm_empty,
        output lsu_sched_drained,
        input  dvstack_wid,
        output dvstack_ptr,
        output mscratch
    );

endinterface
