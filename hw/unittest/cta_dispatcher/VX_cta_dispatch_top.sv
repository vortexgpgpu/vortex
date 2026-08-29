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
    input  wire [`VX_CFG_MEM_ADDR_WIDTH-1:0]   in_param,
    input  wire [`VX_CFG_LMEM_LOG_SIZE:0]      in_lmem_size,
    input  wire [CTA_TID_WIDTH:0]       in_block_size,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_x,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_y,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_z,

    input  wire [`VX_CFG_NUM_WARPS-1:0]        active_warps,
    input  wire                         warp_done,
    input  wire [NW_WIDTH-1:0]          warp_done_wid,

    // Outputs
    output wire                         cta_fire,
    output wire [NW_WIDTH-1:0]          cta_wid
);

    VX_kmu_bus_if kmu_bus();

    // The KMU is not instantiated in this unit test, so the bench supplies the
    // values the dispatcher expects precomputed on the bus:
    //   aligned_lmem_size  — in_lmem_size rounded up to MEM_BLOCK_SIZE.
    //   cluster_size = 1 — degenerate (1,1,1) cluster, so every block is its own
    //   cluster and always first-of-cluster.
    wire [`VX_CFG_LMEM_LOG_SIZE:0] aligned_lmem_size_w =
        ((`VX_CFG_LMEM_LOG_SIZE+1)'(in_lmem_size) + (`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE-1))
        & ~((`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE-1));

    // The bus carries the launch as a flat data beat; the dispatcher casts it
    // back to kmu_req_t. CTA ids are allocated inside the dispatcher.
    kmu_req_t kmu_req;
    always_comb begin
        kmu_req = '0;
        kmu_req.kind              = KMU_KIND_COMPUTE;
        kmu_req.PC                = in_PC;
        kmu_req.entry             = in_PC;
        kmu_req.param             = in_param;
        kmu_req.ctx_id            = '0;
        kmu_req.aligned_lmem_size = aligned_lmem_size_w;
        kmu_req.args.compute.grid_dim[0]  = in_grid_dim_x;
        kmu_req.args.compute.grid_dim[1]  = in_grid_dim_y;
        kmu_req.args.compute.grid_dim[2]  = in_grid_dim_z;
        kmu_req.args.compute.block_idx[0] = in_block_idx_x;
        kmu_req.args.compute.block_idx[1] = in_block_idx_y;
        kmu_req.args.compute.block_idx[2] = in_block_idx_z;
        kmu_req.args.compute.block_dim[0] = in_block_dim_x;
        kmu_req.args.compute.block_dim[1] = in_block_dim_y;
        kmu_req.args.compute.block_dim[2] = in_block_dim_z;
        kmu_req.args.compute.block_size   = in_block_size;
        kmu_req.args.compute.warp_step[0] = in_warp_step_x;
        kmu_req.args.compute.warp_step[1] = in_warp_step_y;
        kmu_req.args.compute.warp_step[2] = in_warp_step_z;
        kmu_req.args.compute.cluster_size = (NW_WIDTH+1)'(1);
        kmu_req.args.compute.is_first_of_cluster = 1'b1;
    end

    assign kmu_bus.valid = task_in_valid;
    assign kmu_bus.kind  = KMU_KIND_COMPUTE;
    assign kmu_bus.eop   = 1'b1;
    assign kmu_bus.dest  = '0;
    assign kmu_bus.data  = kmu_req;
    assign task_in_ready = kmu_bus.ready;
    `UNUSED_VAR (in_cta_id)

    wire [PC_BITS-1:0]      cta_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] cta_tmask;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param;
    wire                    cta_init;
    wire                    busy;

    // CTA-CSR read-back / schedule lookup ports are unexercised by this bench
    // (it only checks warp dispatch via cta_fire/cta_wid), so the read indices
    // are tied off and the read-back outputs are left unused.
    cta_csrs_t              cta_rd_csrs;
    cta_lane_t [`VX_CFG_NUM_THREADS-1:0] cta_rd_lane;
    wire [NCTA_WIDTH-1:0]   schedule_cta_id;

    `UNUSED_VAR (cta_PC)
    `UNUSED_VAR (cta_tmask)
    `UNUSED_VAR (cta_param)
    `UNUSED_VAR (cta_init)
    `UNUSED_VAR (busy)
    `UNUSED_VAR (cta_rd_csrs)
    `UNUSED_VAR (cta_rd_lane)
    `UNUSED_VAR (schedule_cta_id)

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
        .cta_param     (cta_param),
        .cta_init      (cta_init),
        .csr_rd_wid    ('0),
        .csr_rd_cta_id ('0),
        .cta_rd_csrs   (cta_rd_csrs),
        .cta_rd_lane   (cta_rd_lane),
        .schedule_wid  ('0),
        .schedule_cta_id (schedule_cta_id),
        .busy          (busy)
    );

endmodule
