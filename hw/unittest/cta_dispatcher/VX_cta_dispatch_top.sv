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

module VX_cta_dispatch_top import VX_gpu_pkg::*;
(
    input wire                          clk,
    input wire                          reset,

    // KMU bus input (struct fields flattened for C++ testbench driving)
    input  wire                         task_in_valid,
    output wire                         task_in_ready,
    input  wire [PC_BITS-1:0]           in_PC,
    input  wire [31:0]                  in_cta_id,
    input  wire [31:0]                  in_block_idx_x,
    input  wire [31:0]                  in_block_idx_y,
    input  wire [31:0]                  in_block_idx_z,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_x,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_y,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_z,
    input  wire [31:0]                  in_grid_dim_x,
    input  wire [31:0]                  in_grid_dim_y,
    input  wire [31:0]                  in_grid_dim_z,
    input  wire [`MEM_ADDR_WIDTH-1:0]   in_param,
    input  wire [`LMEM_LOG_SIZE:0]      in_lmem_size,
    input  wire [CTA_TID_WIDTH:0]       in_block_size,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_x,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_y,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_z,

    input  wire [`NUM_WARPS-1:0]        active_warps,
    input  wire                         warp_done,
    input  wire [NW_WIDTH-1:0]          warp_done_wid,

    // Outputs
    output wire                         cta_fire,
    output wire [NW_WIDTH-1:0]          cta_wid
);

    VX_kmu_bus_if kmu_bus();

    assign kmu_bus.valid             = task_in_valid;
    assign task_in_ready             = kmu_bus.ready;
    assign kmu_bus.data.PC           = in_PC;
    assign kmu_bus.data.cta_id       = in_cta_id;
    assign kmu_bus.data.block_idx[0] = in_block_idx_x;
    assign kmu_bus.data.block_idx[1] = in_block_idx_y;
    assign kmu_bus.data.block_idx[2] = in_block_idx_z;
    assign kmu_bus.data.block_dim[0] = in_block_dim_x;
    assign kmu_bus.data.block_dim[1] = in_block_dim_y;
    assign kmu_bus.data.block_dim[2] = in_block_dim_z;
    assign kmu_bus.data.grid_dim[0]  = in_grid_dim_x;
    assign kmu_bus.data.grid_dim[1]  = in_grid_dim_y;
    assign kmu_bus.data.grid_dim[2]  = in_grid_dim_z;
    assign kmu_bus.data.param        = in_param;
    assign kmu_bus.data.lmem_size    = in_lmem_size;
    assign kmu_bus.data.block_size   = in_block_size;
    assign kmu_bus.data.warp_step[0] = in_warp_step_x;
    assign kmu_bus.data.warp_step[1] = in_warp_step_y;
    assign kmu_bus.data.warp_step[2] = in_warp_step_z;

    wire [PC_BITS-1:0]      cta_PC;
    wire [`NUM_THREADS-1:0] cta_tmask;
    cta_csrs_t              cta_csrs;
    wire                    cta_init;
    wire                    busy;

    `UNUSED_VAR (cta_PC)
    `UNUSED_VAR (cta_tmask)
    `UNUSED_VAR (cta_csrs)
    `UNUSED_VAR (cta_init)
    `UNUSED_VAR (busy)

    VX_cta_dispatch cta_dispatch (
        .clk           (clk),
        .reset         (reset),
        .kmu_bus_if    (kmu_bus),
        .active_warps  (active_warps),
        .warp_done     (warp_done),
        .warp_done_wid (warp_done_wid),
        .cta_fire      (cta_fire),
        .cta_wid       (cta_wid),
        .cta_PC        (cta_PC),
        .cta_tmask     (cta_tmask),
        .cta_csrs      (cta_csrs),
        .cta_init      (cta_init),
        .busy          (busy)
    );

endmodule
