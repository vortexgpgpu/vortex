`include "VX_define.vh"

module VX_dxa_group_tracker_top #(
    parameter DEPTH = `VX_CFG_DXA_GROUP_DEPTH,
    parameter LIMIT = `VX_CFG_DXA_GROUP_MAX_PENDING,
    parameter GID_W = `CLOG2(DEPTH),
    parameter PTR_W = GID_W + 1,
    parameter COUNT_W = `CLOG2(LIMIT + 1)
) (
    input wire clk,
    input wire reset,
    input wire issue_valid,
    input wire issue_query,
    input wire issue_wid,
    output wire [1:0] issue_result,
    output wire [GID_W-1:0] issue_gid,
    input wire commit_valid,
    input wire commit_wid,
    output wire [1:0] commit_result,
    input wire completion_valid,
    output wire completion_ready,
    input wire completion_wid,
    input wire [GID_W-1:0] completion_gid,
    input wire wq_valid,
    input wire wq_wid,
    input wire [4:0] wq_n,
    output wire wq_satisfied,
    output wire [1:0] unlock_mask,
    input wire owner_close_valid,
    input wire owner_close_wid,
    input wire owner_start_valid,
    input wire owner_start_wid,
    output wire [1:0] drained_mask,
    output wire [2*PTR_W-1:0] head_flat,
    output wire [2*PTR_W-1:0] tail_flat,
    output wire [1:0] obs_open,
    input wire query_wid,
    input wire [GID_W-1:0] query_gid,
    output wire [COUNT_W-1:0] query_pending
);
    wire [PTR_W-1:0] obs_head[2], obs_tail[2];
    wire [DEPTH*COUNT_W-1:0] obs_pending[2];
    assign head_flat = {obs_head[1], obs_head[0]};
    assign tail_flat = {obs_tail[1], obs_tail[0]};
    assign query_pending = obs_pending[query_wid][COUNT_W*int'(query_gid) +: COUNT_W];
    VX_dxa_group_tracker #(.NUM_WARPS(2), .RING_DEPTH(DEPTH), .MAX_PENDING(LIMIT)) dut (.*);
endmodule
