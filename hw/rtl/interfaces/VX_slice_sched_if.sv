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

// Core scheduler state exchanged with one frontend slice's warp scheduler.
// Core-wide events arrive already addressed to the slice: warp ids are
// slice-local (wid = {wfs, fsw}) and per-warp masks hold only its warps.
interface VX_slice_sched_if import VX_gpu_pkg::*; ();

    localparam SW = PER_FETCH_WARPS;
    localparam LW = FETCH_WFS_W;

    // CTA launch
    logic                   cta_fire;
    logic [LW-1:0]          cta_lid;
    logic [PC_BITS-1:0]     cta_PC;
    logic [`VX_CFG_NUM_THREADS-1:0] cta_tmask;
    logic [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param;
    logic                   cta_init;

    // CTA-CSR store writes
    logic                   warp_write;
    logic [LW-1:0]          warp_waddr;
    cta_warp_t              warp_wdata;
    logic                   id_write;
    logic [LW-1:0]          id_waddr;
    logic [NCTA_WIDTH-1:0]  id_wdata;

    // wspawn
    logic                   wspawn_apply;
    logic [SW-1:0]          wspawn_wmask;
    logic [PC_BITS-1:0]     wspawn_pc;
    logic                   wspawn_src;       // the spawning warp is in this slice
    logic [LW-1:0]          wspawn_src_lid;
    logic [`VX_CFG_MEM_ADDR_WIDTH-1:0] wspawn_mscratch;

    // barrier release
    logic                   bar_unlock_valid;
    logic [SW-1:0]          bar_unlock_mask;

    // warp retirement, as the core-wide CTA accounting takes it
    logic                   retire_valid;
    logic [LW-1:0]          retire_lid;

    // every warp of the core has a full ibuffer
    logic                   all_ibuf_full;
`ifdef EXT_SCHED_TIMEOUT_DUMP
    // the core-wide stall timeout is about to fire
    logic                   timeout_dump;
`endif

    // slice status and performance events
    logic [SW-1:0]          active;
    logic [SW-1:0]          stalled;
    logic [SW-1:0]          pending_empty;
    // next-state ibuffer full, registered core-wide into all_ibuf_full
    logic [SW-1:0]          ibuf_full_n;
    logic [FETCH_ALU_PORTS-1:0] branch_valid;
    logic [FETCH_SFU_PORTS-1:0] divergence;
    logic                   sched_valid;
    logic                   sched_fire;
    logic [`VX_CFG_NUM_THREADS-1:0] sched_tmask;

    modport master (
        output cta_fire,
        output cta_lid,
        output cta_PC,
        output cta_tmask,
        output cta_param,
        output cta_init,
        output warp_write,
        output warp_waddr,
        output warp_wdata,
        output id_write,
        output id_waddr,
        output id_wdata,
        output wspawn_apply,
        output wspawn_wmask,
        output wspawn_pc,
        output wspawn_src,
        output wspawn_src_lid,
        output wspawn_mscratch,
        output bar_unlock_valid,
        output bar_unlock_mask,
        output retire_valid,
        output retire_lid,
        output all_ibuf_full,
`ifdef EXT_SCHED_TIMEOUT_DUMP
        output timeout_dump,
`endif
        input  active,
        input  stalled,
        input  ibuf_full_n,
        input  pending_empty,
        input  branch_valid,
        input  divergence,
        input  sched_valid,
        input  sched_fire,
        input  sched_tmask
    );

    modport slave (
        input  cta_fire,
        input  cta_lid,
        input  cta_PC,
        input  cta_tmask,
        input  cta_param,
        input  cta_init,
        input  warp_write,
        input  warp_waddr,
        input  warp_wdata,
        input  id_write,
        input  id_waddr,
        input  id_wdata,
        input  wspawn_apply,
        input  wspawn_wmask,
        input  wspawn_pc,
        input  wspawn_src,
        input  wspawn_src_lid,
        input  wspawn_mscratch,
        input  bar_unlock_valid,
        input  bar_unlock_mask,
        input  retire_valid,
        input  retire_lid,
        input  all_ibuf_full,
`ifdef EXT_SCHED_TIMEOUT_DUMP
        input  timeout_dump,
`endif
        output active,
        output stalled,
        output ibuf_full_n,
        output pending_empty,
        output branch_valid,
        output divergence,
        output sched_valid,
        output sched_fire,
        output sched_tmask
    );

endinterface
