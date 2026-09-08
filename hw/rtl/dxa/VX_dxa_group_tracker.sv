// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

module VX_dxa_group_tracker #(
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,
    parameter RING_DEPTH = `VX_CFG_DXA_GROUP_DEPTH,
    parameter MAX_PENDING = `VX_CFG_DXA_GROUP_MAX_PENDING,
    parameter WID_W = `UP(`CLOG2(NUM_WARPS)),
    parameter GID_W = `CLOG2(RING_DEPTH),
    parameter PTR_W = GID_W + 1,
    parameter REMAIN_W = `CLOG2(MAX_PENDING + 1),
    parameter WAIT_N_W = 5
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
    input wire [WAIT_N_W-1:0] wq_n,
    output wire wq_satisfied,
    output wire [NUM_WARPS-1:0] unlock_mask,
    input wire owner_close_valid,
    input wire [WID_W-1:0] owner_close_wid,
    input wire owner_start_valid,
    input wire [WID_W-1:0] owner_start_wid,
    output wire [NUM_WARPS-1:0] drained_mask,
    output wire [PTR_W-1:0] obs_head [NUM_WARPS],
    output wire [PTR_W-1:0] obs_tail [NUM_WARPS],
    output wire [NUM_WARPS-1:0] obs_open,
    output wire [RING_DEPTH*REMAIN_W-1:0] obs_pending [NUM_WARPS]
);
    `STATIC_ASSERT(RING_DEPTH >= 2 && `IS_POW2(RING_DEPTH), ("DXA group depth must be a power of two >= 2"))
    `STATIC_ASSERT(RING_DEPTH <= (1 << WAIT_N_W), ("DXA group depth exceeds wait immediate range"))
    `STATIC_ASSERT(MAX_PENDING > 0, ("DXA group pending limit must be positive"))

    reg [PTR_W-1:0] head_r [NUM_WARPS], tail_r [NUM_WARPS];
    reg [REMAIN_W-1:0] pending_r [NUM_WARPS][RING_DEPTH];
    reg [NUM_WARPS-1:0] open_r, closed_r, wait_active_r;
    reg [WAIT_N_W-1:0] wait_n_r [NUM_WARPS];

    wire [PTR_W-1:0] issue_depth = tail_r[issue_wid] - head_r[issue_wid];
    assign issue_gid = tail_r[issue_wid][GID_W-1:0];
    wire issue_full = (!open_r[issue_wid] && issue_depth == PTR_W'(RING_DEPTH))
                   || pending_r[issue_wid][issue_gid] == REMAIN_W'(MAX_PENDING);
    assign issue_result = closed_r[issue_wid] ? 2'd2 : issue_full ? 2'd1 : 2'd0;
    assign commit_result = closed_r[commit_wid] ? 2'd2 : 2'd0;
    wire issue_fire = issue_valid && issue_result == 0;
    wire commit_fire = commit_valid && commit_result == 0;
    assign completion_ready = 1'b1;

    wire completion_new_issue = issue_fire && completion_wid == issue_wid && completion_gid == issue_gid;
    wire completion_apply = completion_valid && (pending_r[completion_wid][completion_gid] != 0 || completion_new_issue);
    logic [REMAIN_W-1:0] pending_v [NUM_WARPS][RING_DEPTH];
    logic [PTR_W-1:0] head_v [NUM_WARPS], tail_v [NUM_WARPS];
    logic [NUM_WARPS-1:0] open_v;

    always @(*) begin
        open_v = open_r;
        for (integer w = 0; w < NUM_WARPS; ++w) begin
            head_v[w] = head_r[w];
            tail_v[w] = tail_r[w];
            for (integer g = 0; g < RING_DEPTH; ++g) begin
                pending_v[w][g] = pending_r[w][g];
                if (issue_fire && issue_wid == WID_W'(w) && issue_gid == GID_W'(g))
                    pending_v[w][g] = pending_v[w][g] + REMAIN_W'(1);
                if (completion_apply && completion_wid == WID_W'(w) && completion_gid == GID_W'(g))
                    pending_v[w][g] = pending_v[w][g] - REMAIN_W'(1);
            end
            if (issue_fire && issue_wid == WID_W'(w))
                open_v[w] = 1'b1;
            if (commit_fire && commit_wid == WID_W'(w) && open_r[w]) begin
                tail_v[w] = tail_r[w] + PTR_W'(1);
                open_v[w] = 1'b0;
            end
            if (head_r[w] != tail_r[w] && pending_v[w][head_r[w][GID_W-1:0]] == 0)
                head_v[w] = head_r[w] + PTR_W'(1);
        end
    end

    wire [PTR_W-1:0] query_depth = tail_v[wq_wid] - head_v[wq_wid];
    assign wq_satisfied = 32'(query_depth) <= 32'(wq_n);

    for (genvar w = 0; w < NUM_WARPS; ++w) begin : g_warp
        wire [PTR_W-1:0] next_depth = tail_v[w] - head_v[w];
        `RUNTIME_ASSERT(next_depth <= PTR_W'(RING_DEPTH) && (!open_v[w] || next_depth < PTR_W'(RING_DEPTH)), ("DXA group reservation overflow wid=%0d", w))
        assign unlock_mask[w] = (wait_active_r[w] && 32'(next_depth) <= 32'(wait_n_r[w]))
                              || (wq_valid && wq_wid == WID_W'(w) && wq_satisfied);
        assign drained_mask[w] = head_r[w] == tail_r[w] && (!open_r[w] || pending_r[w][tail_r[w][GID_W-1:0]] == 0);
        assign obs_head[w] = head_r[w];
        assign obs_tail[w] = tail_r[w];
        assign obs_open[w] = open_r[w];
        for (genvar g = 0; g < RING_DEPTH; ++g) begin : g_observe
            assign obs_pending[w][g*REMAIN_W +: REMAIN_W] = pending_r[w][g];
            if (MAX_PENDING < ((1 << REMAIN_W) - 1)) begin : g_check_limit
                `RUNTIME_ASSERT(pending_r[w][g] <= REMAIN_W'(MAX_PENDING), ("DXA group pending overflow wid=%0d gid=%0d", w, g))
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                head_r[w] <= '0;
                tail_r[w] <= '0;
                open_r[w] <= 1'b0;
                closed_r[w] <= 1'b0;
                wait_active_r[w] <= 1'b0;
                wait_n_r[w] <= '0;
                for (integer g = 0; g < RING_DEPTH; ++g)
                    pending_r[w][g] <= '0;
            end else begin
                head_r[w] <= head_v[w];
                tail_r[w] <= tail_v[w];
                open_r[w] <= open_v[w];
                for (integer g = 0; g < RING_DEPTH; ++g)
                    pending_r[w][g] <= pending_v[w][g];
                if (unlock_mask[w])
                    wait_active_r[w] <= 1'b0;
                if (wq_valid && wq_wid == WID_W'(w)) begin
                    wait_active_r[w] <= !wq_satisfied;
                    wait_n_r[w] <= wq_n;
                end
                if (owner_close_valid && owner_close_wid == WID_W'(w)) begin
                    closed_r[w] <= 1'b1;
                    wait_active_r[w] <= 1'b0;
                end
                if (owner_start_valid && owner_start_wid == WID_W'(w) && drained_mask[w]) begin
                    head_r[w] <= '0;
                    tail_r[w] <= '0;
                    open_r[w] <= 1'b0;
                    closed_r[w] <= 1'b0;
                    wait_active_r[w] <= 1'b0;
                end
            end
        end
    end

    // Tags intentionally have no generation: transport must deliver exactly once, and owner reuse must drain.
    `RUNTIME_ASSERT(!completion_valid || completion_apply, ("DXA group completion underflow wid=%0d gid=%0d", completion_wid, completion_gid))
    `RUNTIME_ASSERT(!owner_start_valid || drained_mask[owner_start_wid], ("DXA owner reuse before source drain wid=%0d", owner_start_wid))
    `RUNTIME_ASSERT(!(issue_valid && commit_valid), ("DXA issue and commit must share an ordered control stream"))
    `RUNTIME_ASSERT(!issue_fire || !wait_active_r[issue_wid], ("DXA issue from parked waiter"))
    `RUNTIME_ASSERT(!commit_fire || !wait_active_r[commit_wid], ("DXA commit from parked waiter"))
`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (!reset) begin
            if (issue_fire)
                `TRACE(1, ("%t: GROUP_ISSUE wid=%0d gid=%0d pending=%0d\n", $time, issue_wid, issue_gid, pending_v[issue_wid][issue_gid]))
            if (issue_query && issue_result == 1)
                `TRACE(1, ("%t: GROUP_BACKPRESSURE wid=%0d gid=%0d depth=%0d pending=%0d\n", $time, issue_wid, issue_gid, issue_depth, pending_r[issue_wid][issue_gid]))
            if (commit_fire)
                `TRACE(1, ("%t: GROUP_COMMIT wid=%0d gid=%0d has_ops=%0d\n", $time, commit_wid, tail_r[commit_wid][GID_W-1:0], open_r[commit_wid]))
            if (completion_apply)
                `TRACE(1, ("%t: SOURCE_CONSUMED wid=%0d gid=%0d pending=%0d\n", $time, completion_wid, completion_gid, pending_v[completion_wid][completion_gid]))
            if (wq_valid)
                `TRACE(1, ("%t: WAIT_READ wid=%0d N=%0d depth=%0d pass=%0d\n", $time, wq_wid, wq_n, query_depth, wq_satisfied))
            for (integer w = 0; w < NUM_WARPS; ++w) begin
                if (head_v[w] != head_r[w])
                    `TRACE(1, ("%t: GROUP_SOURCE_RETIRE wid=%0d gid=%0d\n", $time, w, head_r[w][GID_W-1:0]))
                if (wait_active_r[w] && unlock_mask[w])
                    `TRACE(1, ("%t: WAIT_READ_WAKE wid=%0d N=%0d\n", $time, w, wait_n_r[w]))
            end
        end
    end
`else
    `UNUSED_VAR (issue_query)
`endif
endmodule

`endif
