// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
module VX_dxa_group_core_endpoint import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,
    parameter RING_DEPTH = `VX_CFG_DXA_GROUP_DEPTH,
    parameter MAX_PENDING = `VX_CFG_DXA_GROUP_MAX_PENDING,
    parameter WID_W = `UP(`CLOG2(NUM_WARPS)),
    parameter GID_W = `CLOG2(RING_DEPTH),
    parameter PTR_W = GID_W + 1,
    parameter REMAIN_W = `CLOG2(MAX_PENDING + 1)
) (
    input wire clk,
    input wire reset,
    input wire issue_valid,
    input wire issue_query,
    input wire [WID_W-1:0] issue_wid,
    output wire [1:0] issue_result,
    output wire [GID_W-1:0] issue_gid,
    input wire commit_valid,
    input wire [WID_W-1:0] commit_wid,
    output wire [1:0] commit_result,
    input wire completion_valid,
    output wire completion_ready,
    input wire [WID_W-1:0] completion_wid,
    input wire [GID_W-1:0] completion_gid,
    input wire wq_valid,
    input wire [WID_W-1:0] wq_wid,
    input wire [4:0] wq_n,
    output wire wq_satisfied,
    output wire [NUM_WARPS-1:0] unlock_mask,
    input wire owner_close_valid,
    input wire [WID_W-1:0] owner_close_wid,
    input wire owner_start_valid,
    input wire [WID_W-1:0] owner_start_wid,
    output wire [NUM_WARPS-1:0] drained_mask
);
    wire [PTR_W-1:0] obs_head [NUM_WARPS], obs_tail [NUM_WARPS];
    wire [NUM_WARPS-1:0] obs_open;
    wire [RING_DEPTH*REMAIN_W-1:0] obs_pending [NUM_WARPS];
    VX_dxa_group_tracker #(
        .NUM_WARPS (NUM_WARPS),
        .RING_DEPTH (RING_DEPTH),
        .MAX_PENDING (MAX_PENDING)
    ) tracker (.*);
    `UNUSED_VAR (obs_open)
    for (genvar w = 0; w < NUM_WARPS; ++w) begin : g_unused
        `UNUSED_VAR (obs_head[w])
        `UNUSED_VAR (obs_tail[w])
        `UNUSED_VAR (obs_pending[w])
    end
endmodule
`endif
