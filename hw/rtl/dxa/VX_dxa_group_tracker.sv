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

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

// Logical READ-group streams are per issuer warp. Operation contexts live in
// one direct-indexed physical array, but each warp owns a static partition;
// there is no global free-list or cross-warp allocation dependency.
module VX_dxa_group_tracker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_WARPS = 4,
    parameter RING_DEPTH = `VX_CFG_DXA_GROUP_DEPTH,
    parameter NUM_CONTEXTS = `VX_CFG_DXA_GROUP_CONTEXTS,
    parameter CTX_GEN_W = `VX_CFG_DXA_GROUP_CTX_GEN_BITS,
    parameter SEQ_W = 8,
    parameter EPOCH_W = 2,
    parameter WAIT_N_W = 5,
    parameter CNTR_W = 32,
    parameter WID_W = `UP(`CLOG2(NUM_WARPS)),
    parameter CTX_IDX_W = `UP(`CLOG2(NUM_CONTEXTS)),
    parameter OP_ID_W = CTX_IDX_W + CTX_GEN_W,
    parameter REMAIN_W = `CLOG2(NUM_CONTEXTS + 1)
) (
    input wire clk,
    input wire reset,
    input wire assert_on_drop,

    input  wire                 issue_valid,
    input  wire                 issue_query,
    input  wire [WID_W-1:0]     issue_wid,
    output wire [1:0]           issue_result,
    output wire [OP_ID_W-1:0]   issue_op_id,
    input  wire                 commit_valid,
    input  wire [WID_W-1:0]     commit_wid,
    output wire [1:0]           commit_result,

    input  wire                 completion_valid,
    output wire                 completion_ready,
    input  wire [WID_W-1:0]     completion_wid,
    input  wire [EPOCH_W-1:0]   completion_epoch,
    input  wire [SEQ_W-1:0]     completion_seq,
    input  wire [OP_ID_W-1:0]   completion_op,

    input  wire                 wq_valid,
    input  wire [WID_W-1:0]     wq_wid,
    input  wire [WAIT_N_W-1:0]  wq_n,
    output wire                 wq_satisfied,
    output wire [NUM_WARPS-1:0] unlock_mask,

    input  wire                 poison_valid,
    input  wire [WID_W-1:0]     poison_wid,
    input  wire                 epoch_adv_valid,
    input  wire [WID_W-1:0]     epoch_adv_wid,
    output wire [NUM_WARPS-1:0] drained_mask,
    output wire [EPOCH_W-1:0]   epoch [NUM_WARPS],

    output wire [1:0]           sticky_status [NUM_WARPS],
    input  wire                 sticky_clear_valid,
    input  wire [WID_W-1:0]     sticky_clear_wid,
    input  wire [1:0]           sticky_clear_mask,

    output wire [SEQ_W-1:0]     obs_sealed_tail [NUM_WARPS],
    output wire [SEQ_W-1:0]     obs_read_head [NUM_WARPS],
    output wire [REMAIN_W-1:0]  obs_open_ops [NUM_WARPS],
    output wire [CNTR_W-1:0]    obs_stale_drops [NUM_WARPS],
    output wire [CNTR_W-1:0]    obs_duplicate_drops [NUM_WARPS],
    output wire [CNTR_W-1:0]    obs_invalid_drops [NUM_WARPS],
    output wire [RING_DEPTH-1:0]          obs_ring_live [NUM_WARPS],
    output wire [RING_DEPTH*REMAIN_W-1:0] obs_ring_remaining [NUM_WARPS],
    output wire [NUM_CONTEXTS-1:0]        obs_context_live,
    output wire [CNTR_W-1:0]              obs_context_stalls
);
    localparam RING_ADDR_W = `CLOG2(RING_DEPTH);
    localparam CONTEXTS_PER_WARP = NUM_CONTEXTS / NUM_WARPS;
    localparam ISSUE_TRACKED       = 2'd0;
    localparam ISSUE_BACKPRESSURE  = 2'd1;
    localparam ISSUE_IGNORED_POISONED = 2'd2;
    localparam COMMIT_ACCEPTED      = 2'd0;
    localparam COMMIT_BACKPRESSURE  = 2'd1;
    localparam COMMIT_IGNORED_POISONED = 2'd2;

    `STATIC_ASSERT(NUM_CONTEXTS > 0, ("DXA group context pool must be nonempty"))
    `STATIC_ASSERT(`IS_POW2(NUM_CONTEXTS), ("DXA group context pool must be power-of-two"))
    `STATIC_ASSERT(NUM_CONTEXTS >= NUM_WARPS, ("DXA group context pool must cover every warp"))
    `STATIC_ASSERT((NUM_CONTEXTS % NUM_WARPS) == 0,
        ("DXA group contexts must be evenly partitioned by warp"))
    `STATIC_ASSERT(`IS_POW2(CONTEXTS_PER_WARP),
        ("DXA contexts-per-warp must be power-of-two"))
    // The ring pointers are modulo-addressed with CLOG2(RING_DEPTH) bits;
    // depth one would produce a zero-width slice and cannot represent a
    // committed boundary separately from the open group.
    `STATIC_ASSERT(RING_DEPTH >= 2, ("DXA group ring must have at least two slots"))
    `STATIC_ASSERT(`IS_POW2(RING_DEPTH), ("DXA group ring must be power-of-two"))
    `STATIC_ASSERT(RING_DEPTH < (1 << (SEQ_W - 1)), ("DXA group ring exceeds sequence half-window"))
    `STATIC_ASSERT(RING_DEPTH <= (1 << WAIT_N_W), ("DXA group ring exceeds wait-N encoding"))
    `STATIC_ASSERT(CTX_GEN_W >= 4, ("DXA group context generation must be at least 4 bits"))
    `STATIC_ASSERT(OP_ID_W == (CTX_IDX_W + CTX_GEN_W), ("DXA group op-id width mismatch"))

    reg [NUM_CONTEXTS-1:0] ctx_live_r, ctx_seen_r, ctx_done_r;
    reg [CTX_GEN_W-1:0] ctx_gen_r [NUM_CONTEXTS];
    reg [WID_W-1:0] ctx_wid_r [NUM_CONTEXTS];
    reg [EPOCH_W-1:0] ctx_epoch_r [NUM_CONTEXTS];
    reg [SEQ_W-1:0] ctx_seq_r [NUM_CONTEXTS];

    reg [RING_DEPTH-1:0] ring_live_r [NUM_WARPS];
    reg [REMAIN_W-1:0] ring_remaining_r [NUM_WARPS][RING_DEPTH];
    reg [SEQ_W-1:0] sealed_tail_r [NUM_WARPS];
    reg [SEQ_W-1:0] read_head_r [NUM_WARPS];
    reg open_valid_r [NUM_WARPS];
    reg [REMAIN_W-1:0] open_remaining_r [NUM_WARPS];
    reg [EPOCH_W-1:0] epoch_r [NUM_WARPS];
    reg [NUM_WARPS-1:0] poisoned_r;
    reg [1:0] sticky_r [NUM_WARPS];
    reg [NUM_WARPS-1:0] wait_active_r;
    reg [WAIT_N_W-1:0] wait_n_r [NUM_WARPS];
    reg [SEQ_W-1:0] wait_snap_r [NUM_WARPS];
`ifdef PERF_ENABLE
    reg [CNTR_W-1:0] stale_drops_r [NUM_WARPS];
    reg [CNTR_W-1:0] duplicate_drops_r [NUM_WARPS];
    reg [CNTR_W-1:0] invalid_drops_r [NUM_WARPS];
    reg [CNTR_W-1:0] context_stalls_r;
`else
    `UNUSED_VAR (issue_query)
`endif

    logic free_valid;
    logic [CTX_IDX_W-1:0] free_idx;
    always @(*) begin
        free_valid = 1'b0;
        free_idx = '0;
        for (integer i = 0; i < NUM_CONTEXTS; ++i) begin
            // Each warp owns a fixed contiguous context partition. This is
            // equivalent to a single RAM indexed by {issue_wid, local_slot}
            // while preventing an unrelated warp from consuming its slots.
            if (!free_valid
             && (i >= (32'(issue_wid) * CONTEXTS_PER_WARP))
             && (i < ((32'(issue_wid) + 1) * CONTEXTS_PER_WARP))
             && !ctx_live_r[i]) begin
                free_valid = 1'b1;
                free_idx = CTX_IDX_W'(i);
            end
        end
    end

    wire [SEQ_W-1:0] issue_span = sealed_tail_r[issue_wid]
                                - read_head_r[issue_wid];
    wire issue_ring_full = !open_valid_r[issue_wid]
                         && (issue_span >= SEQ_W'(RING_DEPTH));
    assign issue_result = poisoned_r[issue_wid] ? ISSUE_IGNORED_POISONED
                        : (issue_ring_full || !free_valid)
                                                    ? ISSUE_BACKPRESSURE
                                                    : ISSUE_TRACKED;
    wire issue_fire = issue_valid && (issue_result == ISSUE_TRACKED);
    wire [CTX_GEN_W-1:0] issue_gen = ctx_gen_r[free_idx] + CTX_GEN_W'(1);
    assign issue_op_id = {issue_gen, free_idx};

    wire [SEQ_W-1:0] commit_span = sealed_tail_r[commit_wid] - read_head_r[commit_wid];
    // An empty commit is a valid zero-operation boundary when the ring has
    // space.  If the ring is full, it is accepted as a trivially complete
    // no-op (without advancing the sequence); non-empty open groups still
    // require a free sealed row.
    wire commit_ring_full = open_valid_r[commit_wid]
                         && (commit_span >= SEQ_W'(RING_DEPTH));
    assign commit_result = poisoned_r[commit_wid] ? COMMIT_IGNORED_POISONED
                         : commit_ring_full        ? COMMIT_BACKPRESSURE
                                                   : COMMIT_ACCEPTED;
    wire commit_fire = commit_valid && (commit_result == COMMIT_ACCEPTED);

    wire [CTX_IDX_W-1:0] completion_idx = completion_op[CTX_IDX_W-1:0];
    wire [CTX_GEN_W-1:0] completion_gen = completion_op[OP_ID_W-1 -: CTX_GEN_W];
    // A zero-latency source (for example, a cancelled/empty operation in a
    // test endpoint) may return its completion in the same cycle as ISSUE.
    // The physical context is still in its pre-issue state during this
    // combinational evaluation, so explicitly bypass the old ctx_live/seen
    // lookup when the returned token is exactly the token being allocated.
    wire completion_new_issue = completion_valid && issue_fire
        && (completion_wid == issue_wid)
        && (completion_epoch == epoch_r[issue_wid])
        && (completion_seq == sealed_tail_r[issue_wid])
        && (completion_op == issue_op_id);
    wire completion_epoch_stale = completion_valid && !completion_new_issue
        && (completion_epoch != epoch_r[completion_wid]);
    wire completion_gen_stale = completion_valid
        && !completion_new_issue && !completion_epoch_stale
        && ctx_seen_r[completion_idx]
        && (ctx_gen_r[completion_idx] != completion_gen);
    wire completion_stale = !completion_new_issue
                          && (completion_epoch_stale || completion_gen_stale);
    wire completion_owner_match = ctx_seen_r[completion_idx]
        && (ctx_gen_r[completion_idx] == completion_gen)
        && (ctx_wid_r[completion_idx] == completion_wid)
        && (ctx_epoch_r[completion_idx] == completion_epoch)
        && (ctx_seq_r[completion_idx] == completion_seq);
    wire completion_duplicate = completion_valid && !completion_new_issue
        && !completion_stale
        && completion_owner_match && ctx_done_r[completion_idx];
    wire completion_apply_existing = completion_valid && !completion_new_issue
        && !completion_stale && completion_owner_match
        && ctx_live_r[completion_idx] && !ctx_done_r[completion_idx];
    wire completion_apply = completion_new_issue || completion_apply_existing;
    wire [SEQ_W-1:0] completion_span = completion_seq - read_head_r[completion_wid];
    wire completion_is_open = completion_apply
        && ((completion_new_issue && (completion_wid == issue_wid))
         || (open_valid_r[completion_wid]
          && (completion_seq == sealed_tail_r[completion_wid])));
    wire completion_is_sealed = completion_apply && !completion_is_open
        && (completion_span < (sealed_tail_r[completion_wid] - read_head_r[completion_wid]));
    wire [RING_ADDR_W-1:0] completion_slot = completion_seq[RING_ADDR_W-1:0];
    wire completion_target_valid = completion_is_open
        || (completion_is_sealed && ring_live_r[completion_wid][completion_slot]);
    wire completion_count_apply = completion_apply && completion_target_valid;
    wire completion_invalid = completion_valid && !completion_stale
        && !completion_duplicate && !completion_count_apply;
    assign completion_ready = 1'b1;

    logic [REMAIN_W-1:0] open_v [NUM_WARPS];
    logic [SEQ_W-1:0] tail_v [NUM_WARPS];
    logic ring_live_v [NUM_WARPS][RING_DEPTH];
    logic [REMAIN_W-1:0] ring_remaining_v [NUM_WARPS][RING_DEPTH];
    logic head_done_v [NUM_WARPS];
    always @(*) begin
        for (integer w = 0; w < NUM_WARPS; ++w) begin
            open_v[w] = open_remaining_r[w];
            if (completion_count_apply && completion_is_open
             && !completion_new_issue
             && (completion_wid == WID_W'(w)))
                open_v[w] = open_v[w] - REMAIN_W'(1);
            if (issue_fire && (issue_wid == WID_W'(w)))
                open_v[w] = open_v[w] + REMAIN_W'(1);
            if (completion_new_issue && (issue_wid == WID_W'(w)))
                open_v[w] = open_v[w] - REMAIN_W'(1);

            tail_v[w] = sealed_tail_r[w];
            if (commit_fire && (commit_wid == WID_W'(w))
             && (!((!open_valid_r[w])
                 && (commit_span >= SEQ_W'(RING_DEPTH)))))
                tail_v[w] = sealed_tail_r[w] + SEQ_W'(1);

            for (integer i = 0; i < RING_DEPTH; ++i) begin
                ring_live_v[w][i] = ring_live_r[w][i];
                ring_remaining_v[w][i] = ring_remaining_r[w][i];
                if (completion_count_apply && completion_is_sealed
                 && (completion_wid == WID_W'(w))
                 && (completion_slot == RING_ADDR_W'(i)))
                    ring_remaining_v[w][i] = ring_remaining_r[w][i] - REMAIN_W'(1);
                if (commit_fire && (commit_wid == WID_W'(w))
                 && (!((!open_valid_r[w])
                     && (commit_span >= SEQ_W'(RING_DEPTH))))
                 && (sealed_tail_r[w][RING_ADDR_W-1:0] == RING_ADDR_W'(i))) begin
                    ring_live_v[w][i] = 1'b1;
                    ring_remaining_v[w][i] = open_v[w];
                end
            end

            head_done_v[w] = (read_head_r[w] != tail_v[w])
                && ring_live_v[w][read_head_r[w][RING_ADDR_W-1:0]]
                && (ring_remaining_v[w][read_head_r[w][RING_ADDR_W-1:0]] == '0);
        end
    end

    `STATIC_ASSERT(SEQ_W == DXA_GROUP_SEQ_W, ("DXA group sequence width mismatch"))
    `STATIC_ASSERT(WAIT_N_W == 5, ("DXA wait-N width mismatch"))

    assign wq_satisfied = dxa_group_wait_satisfied(
        wq_n, sealed_tail_r[wq_wid], read_head_r[wq_wid], SEQ_W'(RING_DEPTH));
    for (genvar w = 0; w < NUM_WARPS; ++w) begin : g_outputs
        wire parked_unlock = wait_active_r[w]
            && dxa_group_wait_satisfied(
                wait_n_r[w], wait_snap_r[w], read_head_r[w], SEQ_W'(RING_DEPTH));
        assign unlock_mask[w] = parked_unlock
                              || (wq_valid && (wq_wid == WID_W'(w)) && wq_satisfied);

        logic owner_context_live;
        always @(*) begin
            owner_context_live = 1'b0;
            for (integer i = 0; i < NUM_CONTEXTS; ++i) begin
                if (ctx_live_r[i] && (ctx_wid_r[i] == WID_W'(w)))
                    owner_context_live = 1'b1;
            end
        end
        assign drained_mask[w] = (read_head_r[w] == sealed_tail_r[w])
                              && (open_remaining_r[w] == '0)
                              && !owner_context_live;
        assign epoch[w] = epoch_r[w];
        assign sticky_status[w] = sticky_r[w];
        assign obs_sealed_tail[w] = sealed_tail_r[w];
        assign obs_read_head[w] = read_head_r[w];
        assign obs_open_ops[w] = open_remaining_r[w];
    `ifdef PERF_ENABLE
        assign obs_stale_drops[w] = stale_drops_r[w];
        assign obs_duplicate_drops[w] = duplicate_drops_r[w];
        assign obs_invalid_drops[w] = invalid_drops_r[w];
    `else
        assign obs_stale_drops[w] = '0;
        assign obs_duplicate_drops[w] = '0;
        assign obs_invalid_drops[w] = '0;
    `endif
        assign obs_ring_live[w] = ring_live_r[w];
        for (genvar s = 0; s < RING_DEPTH; ++s) begin : g_ring_obs
            assign obs_ring_remaining[w][s*REMAIN_W +: REMAIN_W]
                = ring_remaining_r[w][s];
        end
    end
    assign obs_context_live = ctx_live_r;
`ifdef PERF_ENABLE
    assign obs_context_stalls = context_stalls_r;
`else
    assign obs_context_stalls = '0;
`endif

    always @(posedge clk) begin
        if (reset) begin
            ctx_live_r <= '0;
            ctx_seen_r <= '0;
            ctx_done_r <= '0;
            poisoned_r <= '0;
            wait_active_r <= '0;
        `ifdef PERF_ENABLE
            context_stalls_r <= '0;
        `endif
            for (integer i = 0; i < NUM_CONTEXTS; ++i) begin
                ctx_gen_r[i] <= '0;
                ctx_wid_r[i] <= '0;
                ctx_epoch_r[i] <= '0;
                ctx_seq_r[i] <= '0;
            end
            for (integer w = 0; w < NUM_WARPS; ++w) begin
                ring_live_r[w] <= '0;
                sealed_tail_r[w] <= '0;
                read_head_r[w] <= '0;
                open_valid_r[w] <= 1'b0;
                open_remaining_r[w] <= '0;
                epoch_r[w] <= '0;
                sticky_r[w] <= '0;
                wait_n_r[w] <= '0;
                wait_snap_r[w] <= '0;
            `ifdef PERF_ENABLE
                stale_drops_r[w] <= '0;
                duplicate_drops_r[w] <= '0;
                invalid_drops_r[w] <= '0;
            `endif
                for (integer i = 0; i < RING_DEPTH; ++i)
                    ring_remaining_r[w][i] <= '0;
            end
        end else begin
        `ifdef PERF_ENABLE
            if (issue_query && (issue_result == ISSUE_BACKPRESSURE))
                context_stalls_r <= context_stalls_r + 1'b1;
        `endif

            if (issue_fire) begin
                ctx_live_r[free_idx] <= 1'b1;
                ctx_seen_r[free_idx] <= 1'b1;
                ctx_done_r[free_idx] <= 1'b0;
                ctx_gen_r[free_idx] <= issue_gen;
                ctx_wid_r[free_idx] <= issue_wid;
                ctx_epoch_r[free_idx] <= epoch_r[issue_wid];
                ctx_seq_r[free_idx] <= sealed_tail_r[issue_wid];
            end
            if (completion_count_apply) begin
                ctx_live_r[completion_idx] <= 1'b0;
                ctx_done_r[completion_idx] <= 1'b1;
            end

            for (integer w = 0; w < NUM_WARPS; ++w) begin
                for (integer i = 0; i < RING_DEPTH; ++i) begin
                    if (epoch_adv_valid && (epoch_adv_wid == WID_W'(w))
                     && drained_mask[w]) begin
                        // Sequence state belongs to the CTA lifetime. The
                        // ring is empty by `drained_mask`; clearing rows here
                        // also makes the next epoch start at group sequence 0
                        // instead of carrying an arbitrary modulo value.
                        ring_live_r[w][i] <= 1'b0;
                        ring_remaining_r[w][i] <= '0;
                    end else if (head_done_v[w]
                     && (read_head_r[w][RING_ADDR_W-1:0] == RING_ADDR_W'(i))) begin
                        ring_live_r[w][i] <= 1'b0;
                        ring_remaining_r[w][i] <= '0;
                    end else begin
                        ring_live_r[w][i] <= ring_live_v[w][i];
                        ring_remaining_r[w][i] <= ring_remaining_v[w][i];
                    end
                end
                if (epoch_adv_valid && (epoch_adv_wid == WID_W'(w))
                 && drained_mask[w]) begin
                    sealed_tail_r[w] <= '0;
                    read_head_r[w] <= '0;
                end else begin
                    sealed_tail_r[w] <= tail_v[w];
                    read_head_r[w] <= read_head_r[w] + SEQ_W'(head_done_v[w]);
                end
                if ((epoch_adv_valid && (epoch_adv_wid == WID_W'(w)) && drained_mask[w])
                 || (commit_fire && (commit_wid == WID_W'(w))))
                    open_remaining_r[w] <= '0;
                else
                    open_remaining_r[w] <= open_v[w];

                // ISSUE and COMMIT are mutually exclusive at the integrated
                // front end: both are decoded on the single SFU/DXA control
                // stream, so one warp cannot present both handshakes in one
                // cycle.  A completion may still coincide with COMMIT; the
                // next-state vectors above account for that race.
                if (commit_fire && (commit_wid == WID_W'(w))) begin
                    open_valid_r[w] <= 1'b0;
                end else if (issue_fire && (issue_wid == WID_W'(w))) begin
                    open_valid_r[w] <= 1'b1;
                end

                if (epoch_adv_valid && (epoch_adv_wid == WID_W'(w)) && drained_mask[w]) begin
                    // A completed-but-uncommitted open group is drainable;
                    // clear its marker before the next CTA/epoch can issue.
                    open_valid_r[w] <= 1'b0;
                    epoch_r[w] <= epoch_r[w] + 1'b1;
                    poisoned_r[w] <= 1'b0;
                    wait_active_r[w] <= 1'b0;
                end else begin
                    if (poison_valid && (poison_wid == WID_W'(w)))
                        poisoned_r[w] <= 1'b1;
                    if (unlock_mask[w])
                        wait_active_r[w] <= 1'b0;
                    else if (wq_valid && (wq_wid == WID_W'(w)) && !wq_satisfied) begin
                        wait_active_r[w] <= 1'b1;
                        wait_n_r[w] <= wq_n;
                        wait_snap_r[w] <= sealed_tail_r[w];
                    end
                end

                begin
                    logic [1:0] sticky_n;
                    sticky_n = sticky_r[w];
                    if (sticky_clear_valid && (sticky_clear_wid == WID_W'(w)))
                        sticky_n = sticky_n & ~sticky_clear_mask;
                    if (poison_valid && (poison_wid == WID_W'(w)))
                        sticky_n = sticky_n | 2'b10;
                    if ((completion_duplicate || completion_invalid)
                     && (completion_wid == WID_W'(w)))
                        sticky_n = sticky_n | 2'b01;
                    sticky_r[w] <= sticky_n;
                end

            `ifdef PERF_ENABLE
                if (completion_stale && (completion_wid == WID_W'(w)))
                    stale_drops_r[w] <= stale_drops_r[w] + 1'b1;
                if (completion_duplicate && (completion_wid == WID_W'(w)))
                    duplicate_drops_r[w] <= duplicate_drops_r[w] + 1'b1;
                if (completion_invalid && (completion_wid == WID_W'(w)))
                    invalid_drops_r[w] <= invalid_drops_r[w] + 1'b1;
            `endif
            end
        end
    end

    // The current decode/front-end emits at most one DXA sub-operation per
    // cycle.  Keeping this invariant explicit prevents a future second
    // request path from silently creating an issue/commit ordering hole.
    `RUNTIME_ASSERT(!(issue_valid && commit_valid),
        ("%t: *** dxa-group-tracker: issue and commit in the same cycle", $time))
    `RUNTIME_ASSERT(!completion_apply || completion_target_valid,
        ("%t: *** dxa-group-tracker: completion targeted a retired group", $time))
    `RUNTIME_ASSERT(!completion_count_apply || !completion_is_open
                 || completion_new_issue
                 || (open_remaining_r[completion_wid] != '0),
        ("%t: *** dxa-group-tracker: open count underflow", $time))
    `RUNTIME_ASSERT(!completion_count_apply || !completion_is_sealed
                 || (ring_remaining_r[completion_wid][completion_slot] != '0),
        ("%t: *** dxa-group-tracker: sealed count underflow", $time))
    `RUNTIME_ASSERT(!assert_on_drop || !completion_duplicate,
        ("%t: *** dxa-group-tracker: duplicate completion", $time))
    `RUNTIME_ASSERT(!assert_on_drop || !completion_invalid,
        ("%t: *** dxa-group-tracker: invalid completion", $time))

endmodule

`endif
