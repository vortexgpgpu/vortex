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

`include "VX_cache_define.vh"

// Flush sequencer. Flush tokens (requests with the flush attribute) at any
// input start one walk: every input is blocked, in-flight requests drain and
// every bank writes back and invalidates its lines. With FLUSH_FORWARD, the
// cache then sends one token down each memory port, behind that port's
// write-backs, and waits for every port to answer, so the next level has
// each port's data before it releases that port's token. Finally each token
// visible at the end of the walk is answered locally, one at a time through
// the cache's own response path, and the inputs reopen.
// A token that arrives during the walk is visible only after all of its
// input's earlier requests were accepted, so the walk covered them.
// Bank arbiters that can hold requests apart from their input order
// (XBAR_SIZE != 0: fan-in slices) drain into the banks first, while the
// inputs are held, so no earlier request of a token's input is passed by a
// later one. A request held in a bank's skid register has left its input but
// not reached the bank, so the walk also waits for every skid to empty.
// An L1-only token (any byteen bit set: a next level with wider words sees
// the bit at the token's line offset) is forwarded with its kind. A
// last-level cache that alone serves every memory client is the point of
// coherence for it: once the token is visible, its input's earlier requests
// are inside the cache, ahead of anything the next kernel sends, so it answers
// the token without a walk. Any other last-level cache still walks, since a
// replicated instance or a client reaching memory beside it may hold or need
// lines it has not written back.
// An IO token (the io attribute: a fence ordering only device accesses) walks
// nowhere. Once nothing is held between the cache's inputs and its memory
// ports, on the cacheable and the non-cacheable path alike, it goes down every
// memory port, so its answer proves the earlier IO writes reached memory.
// The L1-only kind splits by what the fence orders. A release token (noalloc
// attribute) owes no invalidation: the banks only quiesce and the token goes
// down to the point of coherence behind the earlier writes. An acquire token
// (io attribute with byte enables set) owes no drain: the cache walks and
// answers, forwarding only to a cache below that is not the point of
// coherence (ACQ_FORWARD).
// Kinds visible together combine to the weakest one that serves them all.
module VX_cache_init import VX_gpu_pkg::*; #(
    parameter NUM_REQS         = 4,              // Number of Word requests per cycle
    parameter NUM_BANKS        = 1,              // Number of banks
    parameter MEM_PORTS        = 1,              // Number of memory ports
    parameter FLUSH_FORWARD    = 1,              // Forward the flush to the next level
    parameter SHARED_LLC       = 0,              // The LLC serving every memory client alone: answers L1-only tokens without a walk
    parameter ACQ_FORWARD      = 0,              // An acquire token must also invalidate the next level
    parameter TAG_WIDTH        = UUID_WIDTH + 1, // core request tag size
    parameter REQ_STAGE        = 0,              // A request stage between the inputs and the banks
    parameter XBAR_SIZE        = 0,              // Requests the bank arbiters can hold apart from input order
    parameter REQ_SEL_WIDTH    = `UP(`CLOG2(NUM_REQS))
) (
    input wire                        clk,
    input wire                        reset,
    // the inputs (VX_cache_req_port): their tokens, and the inputs' gating
    input wire [NUM_REQS-1:0]         flush_req_mask,  // a token is visible at the input
    input wire [NUM_REQS-1:0]         flush_full_mask,
    input wire [NUM_REQS-1:0]         flush_l1_mask,
    input wire [NUM_REQS-1:0]         flush_io_mask,
    input wire [NUM_REQS-1:0]         flush_acq_mask,
    input wire [NUM_REQS-1:0]         flush_rel_mask,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0] req_tag,
    input wire [NUM_REQS-1:0]         rsp_pending,     // the input's token is still to be answered
    output wire                       flush_hold,      // every input closed for a token
    output wire [NUM_REQS-1:0]        dbg_stall,
    output wire                       pending_load,    // the visible tokens become pending
    output wire                       pending_owed,    // only those a quiesce without a walk serves
    output wire [NUM_REQS-1:0]        rsp_take,        // the input's token is answered now
    input wire                        req_stage_empty, // nothing accepted at the outputs short of the banks
    input wire [NUM_REQS-1:0]         xbar_req_fire,   // a request enters the bank arbiters
    input wire [NUM_BANKS-1:0]        bank_req_fire,   // a request enters a bank
    input wire [NUM_BANKS-1:0]        bank_req_held,   // a request sits in a bank's skid register
    output wire [NUM_BANKS-1:0]       flush_begin,
    output wire                       flush_skip,    // the banks only quiesce: no walk is owed
    output wire [`UP(UUID_WIDTH)-1:0] flush_uuid,
    input wire [NUM_BANKS-1:0]        flush_end,
    // flush tokens to the next level
    input wire                        mem_req_empty, // no write-back left between the banks and the ports
    input wire                        nc_req_empty,  // no request held between the inputs and the memory side
    output wire [MEM_PORTS-1:0]       mem_flush_valid,
    output wire                       mem_flush_l1_only,
    output wire                       mem_flush_io,
    output wire                       mem_flush_noalloc,
    input wire [MEM_PORTS-1:0]        mem_flush_ready,
    output wire                       mem_flush_rsp_phase,
    input wire [MEM_PORTS-1:0]        mem_flush_rsp_valid,
    // answers to the requesters' tokens
    output wire                       flush_rsp_valid,
    output wire [REQ_SEL_WIDTH-1:0]   flush_rsp_idx,
    output wire [NUM_REQS-1:0]        flush_rsp_onehot,
    output wire [TAG_WIDTH-1:0]       flush_rsp_tag,
    input wire                        flush_rsp_ready
);
    localparam STATE_IDLE    = 0;
    localparam STATE_WAIT1   = 1;
    localparam STATE_FLUSH   = 2;
    localparam STATE_WAIT2   = 3;
    localparam STATE_FORWARD = 4;
    localparam STATE_RESPOND = 5;

    reg [2:0] state, state_n;

    // track requests accepted at the inputs that have not reached a bank: the
    // request stages, the bank arbiters' slices and the banks' skid registers

    wire stage_empty;
    if (REQ_STAGE != 0) begin : g_req_stage
        assign stage_empty = req_stage_empty;
    end else begin : g_no_req_stage
        assign stage_empty = 1'b1;
        `UNUSED_VAR (req_stage_empty)
    end

    wire xbar_drained;
    if (XBAR_SIZE != 0) begin : g_xbar_inflight

        localparam REQS_W  = `CLOG2(NUM_REQS+1);
        localparam INCRW   = `CLOG2(`MIN(NUM_REQS, XBAR_SIZE)+1);
        localparam BANKS_W = `CLOG2(NUM_BANKS+1);

        // The fires and their counts are registered ahead of the counter, so
        // the arbitration, the popcounts and the counter's add are three
        // cycles apart. A request is in flight from the cycle after it fires:
        // until the counter holds it, the registered fire or count does.
        // Decrements arrive as late, so the counter only overstates.
        reg [NUM_REQS-1:0]  xbar_req_fire_r;
        reg [NUM_BANKS-1:0] bank_req_fire_r;
        reg [REQS_W-1:0]    xbar_req_cnt_r;
        reg [BANKS_W-1:0]   bank_req_cnt_r;
        wire [REQS_W-1:0]   xbar_req_cnt;
        wire [BANKS_W-1:0]  bank_req_cnt;

        `POP_COUNT(xbar_req_cnt, xbar_req_fire_r);
        `POP_COUNT(bank_req_cnt, bank_req_fire_r);

        always @(posedge clk) begin
            if (reset) begin
                xbar_req_fire_r <= '0;
                bank_req_fire_r <= '0;
                xbar_req_cnt_r  <= '0;
                bank_req_cnt_r  <= '0;
            end else begin
                xbar_req_fire_r <= xbar_req_fire;
                bank_req_fire_r <= bank_req_fire;
                xbar_req_cnt_r  <= xbar_req_cnt;
                bank_req_cnt_r  <= bank_req_cnt;
            end
        end

        wire xbar_empty;

        VX_pending_size #(
            .SIZE  (XBAR_SIZE),
            .INCRW (INCRW),
            .DECRW (BANKS_W)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (INCRW'(xbar_req_cnt_r)),
            .decr  (bank_req_cnt_r),
            .empty (xbar_empty),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign xbar_drained = xbar_empty && ~(| xbar_req_fire_r) && (xbar_req_cnt_r == '0);

    end else begin : g_no_xbar_inflight
        assign xbar_drained = 1'b1;
        `UNUSED_VAR ({xbar_req_fire, bank_req_fire})
    end

    // a token waits for that drain where any of it exists
    wire drain_wait = (REQ_STAGE != 0) || (XBAR_SIZE != 0) || (| bank_req_held);
    wire no_inflight_reqs = stage_empty && xbar_drained && ~(| bank_req_held);

    wire flush_req_enable = (| flush_req_mask);
    // Over many inputs the hold of every input on any input's token is
    // registered: the other inputs close a cycle after a token shows (they are
    // not ordered against it, and the walk waits for what they sent), while a
    // token's own input never passes it.
    localparam HOLD_REG = (NUM_REQS > 4);
    if (HOLD_REG) begin : g_hold_reg
        reg flush_hold_r;
        always @(posedge clk) begin
            if (reset) begin
                flush_hold_r <= 1'b0;
            end else begin
                flush_hold_r <= flush_req_enable;
            end
        end
        assign flush_hold = flush_hold_r;
    end else begin : g_hold_now
        assign flush_hold = flush_req_enable;
    end

    wire any_full = (| flush_full_mask);
    wire any_l1   = (| flush_l1_mask);
    wire any_io   = (| flush_io_mask);
    wire any_acq  = (| flush_acq_mask);
    wire any_rel  = (| flush_rel_mask);
    // a walk is owed to a full token, and to an L1-only or acquire one above
    // the point of coherence
    wire flush_walk = any_full || ((any_l1 || any_acq) && (SHARED_LLC == 0));
    // a drain with no walk is owed to an IO token, and to a release one above
    // the point of coherence
    wire flush_drain = any_io || (any_rel && (SHARED_LLC == 0));

    reg [NUM_BANKS-1:0] flush_done, flush_done_n;
    reg                 fwd_go, fwd_go_n;   // the crossbar drained; only tokens enter it now
    reg [MEM_PORTS-1:0] fwd_sent, fwd_sent_n;
    reg [MEM_PORTS-1:0] fwd_done, fwd_done_n;
    reg                 pending_load_n, pending_owed_n;
    reg [`UP(UUID_WIDTH)-1:0] flush_uuid_r, flush_uuid_n;
    reg                 l1_only_r, l1_only_n;
    reg                 io_r, io_n;
    reg                 noalloc_r, noalloc_n;
    reg                 skip_r, skip_n;

    // answer the lowest pending input; its token stays at the input until then
    wire [NUM_REQS-1:0]      rsp_sel_onehot;
    wire [REQ_SEL_WIDTH-1:0] rsp_sel_idx;
    wire                     rsp_sel_valid;
    VX_priority_encoder #(
        .N (NUM_REQS)
    ) rsp_sel (
        .data_in    (rsp_pending),
        .onehot_out (rsp_sel_onehot),
        .index_out  (rsp_sel_idx),
        .valid_out  (rsp_sel_valid)
    );
    assign flush_rsp_valid = (state == STATE_RESPOND) && rsp_sel_valid;
    assign flush_rsp_idx   = rsp_sel_idx;
    assign flush_rsp_onehot = rsp_sel_onehot;
    wire flush_rsp_fire = flush_rsp_valid && flush_rsp_ready;

    VX_onehot_mux #(
        .DATAW (TAG_WIDTH),
        .N     (NUM_REQS)
    ) rsp_tag_mux (
        .data_in  (req_tag),
        .sel_in   (rsp_sel_onehot),
        .data_out (flush_rsp_tag)
    );

`ifdef DBG_FLUSH_STALL
    // Random input backpressure, a new mask every 128 cycles (each input
    // stalled half the time), to widen ordering windows under test.
    reg [31:0] dbg_lfsr_r;
    reg [6:0]  dbg_cnt_r;
    reg [NUM_REQS-1:0] dbg_stall_r;
    always @(posedge clk) begin
        if (reset) begin
            dbg_lfsr_r  <= 32'hACE1_2468;
            dbg_cnt_r   <= '0;
            dbg_stall_r <= '0;
        end else begin
            dbg_lfsr_r <= {dbg_lfsr_r[30:0], dbg_lfsr_r[31] ^ dbg_lfsr_r[21] ^ dbg_lfsr_r[1] ^ dbg_lfsr_r[0]};
            dbg_cnt_r  <= dbg_cnt_r + 7'd1;
            if (dbg_cnt_r == '0) begin
                dbg_stall_r <= (NUM_REQS > 1) ? NUM_REQS'(dbg_lfsr_r) : '0;
            end
        end
    end
    assign dbg_stall = dbg_stall_r;
`else
    assign dbg_stall = '0;
`endif

    // the token answered now: its input consumes it (VX_cache_req_port)
    assign rsp_take = rsp_sel_onehot & {NUM_REQS{flush_rsp_fire}};

    wire [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] core_bus_in_uuid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_uuid
        if (UUID_WIDTH != 0) begin : g_uuid
            assign core_bus_in_uuid[i] = req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin : g_no_uuid
            assign core_bus_in_uuid[i] = 0;
        end
    end

    // an IO token also waits for the non-cacheable path, which carries the IO
    // writes it orders
    wire fwd_ready = mem_req_empty && (~io_r || l1_only_r || nc_req_empty);

    // The kind forwarded after a walk: full if a full token or an IO one asks
    // for more than the point of coherence, L1-only if any token owes a drain
    // to it, acquire if only acquire tokens were visible.
    wire walk_fwd_acq = any_acq && ~any_full && ~any_io && ~any_l1 && ~any_rel;
    wire walk_fwd_l1  = (any_l1 || any_rel) && ~any_full && ~any_io;

    always @(*) begin
        state_n       = state;
        flush_done_n  = flush_done;
        fwd_go_n      = fwd_go;
        fwd_sent_n    = fwd_sent;
        fwd_done_n    = fwd_done;
        pending_load_n = 1'b0;
        pending_owed_n = 1'b0;
        flush_uuid_n  = flush_uuid_r;
        l1_only_n     = l1_only_r;
        io_n          = io_r;
        noalloc_n     = noalloc_r;
        skip_n        = skip_r;
        case (state)
            // STATE_IDLE:
            default: begin
                if (flush_hold && flush_req_enable) begin
                    if (drain_wait) begin
                        state_n = STATE_WAIT1;
                    end else if (~flush_walk && ~flush_drain) begin
                        pending_load_n = 1'b1;
                        state_n = STATE_RESPOND;
                    end else begin
                        skip_n = ~flush_walk;
                        state_n = STATE_FLUSH;
                    end
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (flush_req_mask[i]) begin
                            flush_uuid_n = core_bus_in_uuid[i];
                        end
                    end
                end
            end
            STATE_WAIT1: begin
                if (no_inflight_reqs) begin
                    if (~flush_walk && ~flush_drain) begin
                        pending_load_n = 1'b1;
                        state_n = STATE_RESPOND;
                    end else begin
                        skip_n = ~flush_walk;
                        state_n = STATE_FLUSH;
                    end
                end
            end
            STATE_FLUSH: begin
                // generate a flush request pulse
                state_n = STATE_WAIT2;
            end
            STATE_WAIT2: begin
                // wait for all banks to finish flushing
                flush_done_n = flush_done | flush_end;
                if (flush_done_n == {NUM_BANKS{1'b1}}) begin
                    flush_done_n = '0;
                    // answer the tokens visible now; later ones start a new walk.
                    // A quiesce without a walk serves only the tokens owing none,
                    // and forwards IO if it serves an IO token, else release.
                    pending_load_n = 1'b1;
                    pending_owed_n = skip_r;
                    l1_only_n = skip_r ? ~any_io : (walk_fwd_l1 || walk_fwd_acq);
                    io_n      = skip_r ? any_io : walk_fwd_acq;
                    noalloc_n = skip_r && ~any_io;
                    // an acquire walk owes the next level nothing unless it too
                    // may hold stale lines
                    state_n = (FLUSH_FORWARD && (skip_r || ~walk_fwd_acq || ACQ_FORWARD)) ? STATE_FORWARD : STATE_RESPOND;
                end
            end
            STATE_FORWARD: begin
                fwd_go_n   = fwd_go | fwd_ready;
                fwd_sent_n = fwd_sent | (mem_flush_valid & mem_flush_ready);
                fwd_done_n = fwd_done | mem_flush_rsp_valid;
                if (fwd_done_n == {MEM_PORTS{1'b1}}) begin
                    fwd_go_n   = 1'b0;
                    fwd_sent_n = '0;
                    fwd_done_n = '0;
                    state_n = STATE_RESPOND;
                end
            end
            STATE_RESPOND: begin
                if ((rsp_pending & ~rsp_take) == '0) begin
                    state_n = STATE_IDLE;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state       <= STATE_IDLE;
            flush_done  <= '0;
            fwd_go      <= 1'b0;
            fwd_sent    <= '0;
            fwd_done    <= '0;
        end else begin
            state       <= state_n;
            flush_done  <= flush_done_n;
            fwd_go      <= fwd_go_n;
            fwd_sent    <= fwd_sent_n;
            fwd_done    <= fwd_done_n;
        end
        flush_uuid_r <= flush_uuid_n;
        l1_only_r    <= l1_only_n;
        io_r         <= io_n;
        noalloc_r    <= noalloc_n;
        skip_r       <= skip_n;
    end

    assign pending_load = pending_load_n;
    assign pending_owed = pending_owed_n;

    assign flush_begin = {NUM_BANKS{state == STATE_FLUSH}};
    assign flush_skip  = skip_r;
    assign flush_uuid = flush_uuid_r;

    // The drain counter counts the tokens too, so the first one sent must not
    // take the others' valid away.
    assign mem_flush_valid     = {MEM_PORTS{(state == STATE_FORWARD) && (fwd_go || fwd_ready)}} & ~fwd_sent;
    assign mem_flush_rsp_phase = (state == STATE_FORWARD);
    assign mem_flush_l1_only   = l1_only_r;
    assign mem_flush_io        = io_r;
    assign mem_flush_noalloc   = noalloc_r;

endmodule
