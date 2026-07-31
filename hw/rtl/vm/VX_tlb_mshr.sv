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

// Shared TLB miss station. Misses on the same VPN dedup onto one entry and one
// walk; a per-entry queue holds an opaque token per waiting requester, and the
// entry drains one token per cycle when the fill lands.
//
// NUM_REQS lanes probe the table in parallel for hit-under-miss; the allocate
// path reuses the probing lane's result (alloc_vpn == probe_vpn[alloc_lane]) so
// the wide dedup compare stays off the enqueue path. A same-VPN request joins
// its entry through the fill: queuing behind the entry's parked requests keeps
// per-lane program order for same-page accesses, which the requester relies on.
// DEDUP_LIVE_EXCLUDES_FAULT drops faulted entries from dedup (a new same-VPN
// request re-walks); with it clear the request joins and shares the fault.
//
// The associative state (valid/vpn/control) is held in flip-flops — an all-entry
// parallel compare cannot come from an addressable RAM. The wide single-read
// payloads (the fill result and the pooled requester tokens) live in VX_dp_ram
// with a registered read at every size: the drain selection is computed one
// cycle ahead from the next-state pending bits and presented as the read
// address, so the data lands exactly when its entry is selected. One
// size-independent timing contract; the primitive is free to map to
// distributed or block RAM.
module VX_tlb_mshr import VX_tlb_pkg::*; #(
    parameter NUM_REQS   = 1,
    parameter MSHR_SIZE  = 4,
    parameter QDATA_W    = 1,
    parameter QDEPTH     = 2,
    parameter DEDUP_LIVE_EXCLUDES_FAULT = 0,
    parameter ID_WIDTH   = `CLOG2(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

    // Per-lane VPN probe: does the VPN match a dedup-live entry?
    input  wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0] probe_vpn,
    output wire [NUM_REQS-1:0]                     probe_match,

    // Allocate / append (at most one per cycle).
    input  wire                             alloc_valid,
    input  wire [TLB_VPN_WIDTH-1:0]         alloc_vpn,
    input  tlb_access_e                     alloc_access,
    input  wire                             alloc_amo,
    input  wire [`UP(`CLOG2(NUM_REQS))-1:0] alloc_lane,
    input  wire [QDATA_W-1:0]               alloc_qdata,
    output wire                             alloc_ready,

    // Walk issue.
    output wire                       issue_valid,
    output wire [`UP(ID_WIDTH)-1:0]   issue_slot,
    output tlb_access_e               issue_access,
    output wire                       issue_amo,
    output wire [TLB_VPN_WIDTH-1:0]   issue_vpn,
    input  wire                       issue_ready,

    // Walk fill (always accepted: the issuing entry is its own landing slot).
    input  wire                       fill_valid,
    input  wire [`UP(ID_WIDTH)-1:0]   fill_slot,
    input  wire                       fill_fault,
    input  wire [TLB_LEVEL_WIDTH-1:0] fill_level,
    input  wire [TLB_PPN_WIDTH-1:0]   fill_ppn,
    input  wire [TLB_FLAGS_WIDTH-1:0] fill_flags,
    output wire                       fill_ready,

    // Array install (non-faulting fill).
    output wire                       install_valid,
    output tlb_entry_t                install_entry,

    // Fill-time fault sideband.
    output wire                       fault_valid,
    output wire [TLB_VPN_WIDTH-1:0]   fault_vpn,
    output tlb_access_e               fault_access,

    // Drain: one queued token per cycle. drain_fault marks a faulted walk's token.
    output wire                       drain_valid,
    output wire [QDATA_W-1:0]         drain_qdata,
    output wire                       drain_fault,
    output wire [TLB_PPN_WIDTH-1:0]   drain_ppn,
    output wire [TLB_LEVEL_WIDTH-1:0] drain_level,
    output wire [TLB_FLAGS_WIDTH-1:0] drain_flags,
    input  wire                       drain_ready,

    input  wire                       flush,
    output wire                       empty
);
    localparam IDX_W     = `CLOG2(MSHR_SIZE);
    localparam SIZE_W    = `CLOG2(QDEPTH + 1);
    localparam PAYLOAD_W = TLB_PPN_WIDTH + TLB_LEVEL_WIDTH + TLB_FLAGS_WIDTH;

    `STATIC_ASSERT((MSHR_SIZE >= 2), ("VX_tlb_mshr requires MSHR_SIZE >= 2"))
    `STATIC_ASSERT(((MSHR_SIZE & (MSHR_SIZE-1)) == 0), ("VX_tlb_mshr requires MSHR_SIZE power-of-two"))
    `STATIC_ASSERT((ID_WIDTH >= IDX_W), ("VX_tlb_mshr requires ID_WIDTH >= CLOG2(MSHR_SIZE)"))

    reg [MSHR_SIZE-1:0]        valid_r;
    reg [MSHR_SIZE-1:0]        issued_r;
    reg [MSHR_SIZE-1:0]        filled_r;
    reg [MSHR_SIZE-1:0]        fault_r;
    reg [TLB_VPN_WIDTH-1:0]    vpn_r    [MSHR_SIZE];
    tlb_access_e               access_r [MSHR_SIZE];
    reg [MSHR_SIZE-1:0]        amo_r;

    // ---------------------------------------------------------------------
    // VPN probes and allocate target
    // ---------------------------------------------------------------------
    // Dedup-live entries: a filled entry stays joinable until it frees — a
    // same-page request must queue behind the entry's parked requests so it
    // cannot overtake an older same-lane access on the hit path (per-lane
    // program order). DEDUP_LIVE_EXCLUDES_FAULT drops only faulted entries,
    // whose requesters are killed rather than replayed: a new request re-walks.
    wire [MSHR_SIZE-1:0] live = DEDUP_LIVE_EXCLUDES_FAULT ? (valid_r & ~fault_r) : valid_r;

    wire [NUM_REQS-1:0][IDX_W-1:0] probe_slot;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_probe
        wire [MSHR_SIZE-1:0] m;
        for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_m
            assign m[e] = live[e] && (vpn_r[e] == probe_vpn[l]);
        end
        VX_priority_encoder #(
            .N (MSHR_SIZE)
        ) probe_enc (
            .data_in    (m),
            `UNUSED_PIN (onehot_out),
            .index_out  (probe_slot[l]),
            .valid_out  (probe_match[l])
        );
    end

    // Allocate reuses the probing lane's dedup result.
    wire             has_match  = probe_match[alloc_lane];
    wire [IDX_W-1:0] match_slot = probe_slot[alloc_lane];

    wire             has_free;
    wire [IDX_W-1:0] free_slot;
    VX_priority_encoder #(
        .N (MSHR_SIZE)
    ) free_enc (
        .data_in    (~valid_r),
        `UNUSED_PIN (onehot_out),
        .index_out  (free_slot),
        .valid_out  (has_free)
    );

    wire [IDX_W-1:0] alloc_target = has_match ? match_slot : free_slot;

    // ---------------------------------------------------------------------
    // Per-entry requester queues, pooled into one shared payload RAM. Entry e
    // owns pool slots [e*QDEPTH, (e+1)*QDEPTH); head/tail/count live in FF. At
    // most one push and one pop per cycle, and within an entry head != tail for
    // 0 < count < QDEPTH, so the two ports never collide -> a single VX_dp_ram
    // serves the whole pool.
    // ---------------------------------------------------------------------
    `STATIC_ASSERT((QDEPTH >= 2), ("VX_tlb_mshr requires QDEPTH >= 2"))
    `STATIC_ASSERT(((QDEPTH & (QDEPTH-1)) == 0), ("VX_tlb_mshr requires QDEPTH power-of-two"))
    localparam PW         = `CLOG2(QDEPTH);
    localparam POOL_DEPTH = MSHR_SIZE * QDEPTH;
    localparam POOL_AW    = IDX_W + PW;

    reg [PW-1:0]     head_r  [MSHR_SIZE];
    reg [PW-1:0]     tail_r  [MSHR_SIZE];
    reg [SIZE_W-1:0] count_r [MSHR_SIZE];

    wire [MSHR_SIZE-1:0] q_empty;
    wire [MSHR_SIZE-1:0] q_full;
    wire [MSHR_SIZE-1:0][SIZE_W-1:0] q_size;
    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_qstat
        assign q_empty[e] = (count_r[e] == '0);
        assign q_full[e]  = (count_r[e] == SIZE_W'(QDEPTH));
        assign q_size[e]  = count_r[e];
    end

    wire [MSHR_SIZE-1:0] q_push;
    wire [MSHR_SIZE-1:0] q_pop;

    // ---------------------------------------------------------------------
    // Allocate accept. Refused during a flush: all state clears at this edge,
    // so an accepted request would be dropped with no replay or kill; the
    // requester retries against the flushed TLB instead.
    // ---------------------------------------------------------------------
    assign alloc_ready = !flush && (has_match ? !q_full[match_slot] : has_free);
    wire alloc_fire = alloc_valid && alloc_ready;

    // ---------------------------------------------------------------------
    // Issue
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] issue_pend = valid_r & ~issued_r;
    wire             has_issue;
    wire [IDX_W-1:0] issue_sel;
    VX_priority_encoder #(
        .N (MSHR_SIZE)
    ) issue_enc (
        .data_in    (issue_pend),
        `UNUSED_PIN (onehot_out),
        .index_out  (issue_sel),
        .valid_out  (has_issue)
    );

    assign issue_valid  = has_issue;
    assign issue_slot   = `UP(ID_WIDTH)'(issue_sel);
    assign issue_access = access_r[issue_sel];
    assign issue_amo    = amo_r[issue_sel];
    assign issue_vpn    = vpn_r[issue_sel];
    wire issue_fire = issue_valid && issue_ready;

    // ---------------------------------------------------------------------
    // Fill
    // ---------------------------------------------------------------------
    assign fill_ready = 1'b1;
    wire fill_fire = fill_valid && fill_ready;
    wire [IDX_W-1:0] fill_idx = fill_slot[IDX_W-1:0];
    `UNUSED_VAR (fill_slot)

    assign install_valid = fill_fire && !fill_fault;
    assign install_entry = '{
        level: fill_level,
        vpn:   vpn_r[fill_idx],
        ppn:   fill_ppn,
        flags: fill_flags
    };

    assign fault_valid  = fill_fire && fill_fault;
    assign fault_vpn    = vpn_r[fill_idx];
    assign fault_access = access_r[fill_idx];

    // ---------------------------------------------------------------------
    // Drain slot selection, computed one cycle ahead so both payload reads
    // use registered addresses with no added drain latency. The selection
    // vector is registered pending state plus the landing fill only — an
    // entry freeing this cycle is deliberately NOT subtracted, keeping the
    // allocate/ready cone out of the read-address path; a just-freed entry
    // can be selected for one dead cycle, where its empty queue (or its
    // re-allocated filled=0 state) gates the drain. The search starts at a
    // pointer that rotates past each popped entry, so an entry fed a steady
    // join stream cannot pin the selection and starve other filled entries.
    // ---------------------------------------------------------------------
    wire                 drain_free;
    wire                 drain_pop;
    wire [MSHR_SIZE-1:0] fill_mask;

    reg [IDX_W-1:0] drain_sel_r;
    reg             has_drain_r;
    reg [IDX_W-1:0] drain_rr_r;

    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_fill_mask
        assign fill_mask[e] = fill_fire && (fill_idx == IDX_W'(e));
    end

    wire [MSHR_SIZE-1:0] drain_pend_n = (valid_r & filled_r) | fill_mask;

    // Rotate the pending vector to the registered pointer: the pointer lags
    // its pop by one cycle, keeping the ready path out of the read-address
    // cone. An entry can win at most two consecutive pops, so the wait bound
    // doubles but stays bounded. Power-of-two MSHR_SIZE makes the index
    // addition below wrap for free.
    wire [2*MSHR_SIZE-1:0] pend_dup = {drain_pend_n, drain_pend_n} >> drain_rr_r;
    wire [MSHR_SIZE-1:0]   pend_rot = pend_dup[MSHR_SIZE-1:0];
    `UNUSED_VAR (pend_dup)

    wire             has_drain_n;
    wire [IDX_W-1:0] sel_rot;
    VX_priority_encoder #(
        .N (MSHR_SIZE)
    ) drain_enc (
        .data_in    (pend_rot),
        `UNUSED_PIN (onehot_out),
        .index_out  (sel_rot),
        .valid_out  (has_drain_n)
    );
    wire [IDX_W-1:0] drain_sel_n = sel_rot + drain_rr_r;

    always @(posedge clk) begin
        if (reset || flush) begin
            drain_sel_r <= '0;
            has_drain_r <= 1'b0;
            drain_rr_r  <= '0;
        end else begin
            drain_sel_r <= drain_sel_n;
            has_drain_r <= has_drain_n;
            if (drain_pop) begin
                drain_rr_r <= drain_sel_r + IDX_W'(1);
            end
        end
    end

    wire drain_ne = !q_empty[drain_sel_r];

    // ---------------------------------------------------------------------
    // Requester payload pool: one VX_dp_ram shared by all entries, addressed
    // {entry, position}. Push writes the tail slot; the read presents the
    // next-cycle head — pre-incremented past a same-cycle pop — so the
    // registered read streams one token per cycle. The count==1 pop+push case
    // reads the slot written this cycle; the write-first read covers it.
    // ---------------------------------------------------------------------
    wire [POOL_AW-1:0] pool_waddr = {alloc_target, tail_r[alloc_target]};
    wire [PW-1:0] pool_rptr_n = head_r[drain_sel_n] + (q_pop[drain_sel_n] ? PW'(1) : PW'(0));
    wire [POOL_AW-1:0] pool_raddr = {drain_sel_n, pool_rptr_n};
    wire [QDATA_W-1:0] pool_rdata;

    VX_dp_ram #(
        .DATAW    (QDATA_W),
        .SIZE     (POOL_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) reqpool (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (alloc_fire),
        .wren  (1'b1),
        .waddr (pool_waddr),
        .wdata (alloc_qdata),
        .raddr (pool_raddr),
        .rdata (pool_rdata)
    );

    // ---------------------------------------------------------------------
    // Fill-result payload store: {ppn, level, flags}, written at fill and read
    // at drain. The read presents the next-cycle selection, so a fill's data
    // is registered during the same cycle its entry becomes drain-pending.
    // ---------------------------------------------------------------------
    wire [PAYLOAD_W-1:0] payload_wdata = {fill_ppn, fill_level, fill_flags};
    wire [PAYLOAD_W-1:0] payload_rdata;

    VX_dp_ram #(
        .DATAW    (PAYLOAD_W),
        .SIZE     (MSHR_SIZE),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) payload_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (fill_fire),
        .wren  (1'b1),
        .waddr (fill_idx),
        .wdata (payload_wdata),
        .raddr (drain_sel_n),
        .rdata (payload_rdata)
    );

    assign {drain_ppn, drain_level, drain_flags} = payload_rdata;

    // ---------------------------------------------------------------------
    // Drain
    // ---------------------------------------------------------------------
    assign drain_valid = has_drain_r && drain_ne;
    assign drain_fault = fault_r[drain_sel_r];
    assign drain_qdata = pool_rdata;
    assign drain_pop   = drain_valid && drain_ready;

    // ---------------------------------------------------------------------
    // Queue push / pop wiring
    // ---------------------------------------------------------------------
    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_qctrl
        assign q_push[e] = alloc_fire && (alloc_target == IDX_W'(e));
        assign q_pop[e]  = drain_pop && (drain_sel_r == IDX_W'(e));
    end

    // Free the drained entry once its queue empties with no concurrent push.
    wire drain_push = alloc_fire && (alloc_target == drain_sel_r);
    assign drain_free = has_drain_r
        && ((drain_ne && drain_pop && (q_size[drain_sel_r] == SIZE_W'(1)) && !drain_push)
         || (!drain_ne && !drain_push));

    // ---------------------------------------------------------------------
    // State update
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset || flush) begin
            valid_r  <= '0;
            issued_r <= '0;
            filled_r <= '0;
            fault_r  <= '0;
            amo_r    <= '0;
        end else begin
            if (alloc_fire && !has_match) begin
                valid_r[free_slot]  <= 1'b1;
                issued_r[free_slot] <= 1'b0;
                filled_r[free_slot] <= 1'b0;
                fault_r[free_slot]  <= 1'b0;
                vpn_r[free_slot]    <= alloc_vpn;
                access_r[free_slot] <= alloc_access;
                amo_r[free_slot]    <= alloc_amo;
            end
            if (issue_fire) begin
                issued_r[issue_sel] <= 1'b1;
            end
            if (fill_fire) begin
                filled_r[fill_idx] <= 1'b1;
                fault_r[fill_idx]  <= fill_fault;
            end
            if (drain_free) begin
                valid_r[drain_sel_r]  <= 1'b0;
                filled_r[drain_sel_r] <= 1'b0;
            end
        end
    end

    // Pool ring pointers. QDEPTH is a power of two, so tail/head wrap for free.
    always @(posedge clk) begin
        if (reset || flush) begin
            for (int e = 0; e < MSHR_SIZE; ++e) begin
                head_r[e]  <= '0;
                tail_r[e]  <= '0;
                count_r[e] <= '0;
            end
        end else begin
            for (int e = 0; e < MSHR_SIZE; ++e) begin
                if (q_push[e]) begin
                    tail_r[e] <= tail_r[e] + PW'(1);
                end
                if (q_pop[e]) begin
                    head_r[e] <= head_r[e] + PW'(1);
                end
                if (q_push[e] && !q_pop[e]) begin
                    count_r[e] <= count_r[e] + SIZE_W'(1);
                end
                if (q_pop[e] && !q_push[e]) begin
                    count_r[e] <= count_r[e] - SIZE_W'(1);
                end
            end
        end
    end

    assign empty = ~(| valid_r);

endmodule
