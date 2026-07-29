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
// entry drains one token per cycle when the fill lands. The L1 and L2 TLBs wrap
// this with their own inbound (lane park vs client alloc) and outbound (replay/
// kill vs client response) glue.
//
// NUM_REQS lanes probe the table in parallel for hit-under-miss; the allocate
// path reuses the probing lane's result (alloc_vpn == probe_vpn[alloc_lane]) so
// the wide dedup compare stays off the enqueue path. DEDUP_LIVE_EXCLUDES_FAULT
// drops faulted entries from dedup (L1: a new same-VPN request re-walks; L2: it
// joins the faulting entry and shares the fault).
//
// The associative state (valid/vpn/control) is held in flip-flops — an all-entry
// parallel compare cannot come from an addressable RAM. The wide fill-result
// payload {ppn,level,flags} is single-read (at drain), so it lives in a VX_dp_ram
// that FORCE_BRAM sizes automatically: distributed RAM (combinational read) at
// small depth, block RAM (registered read) once a large config makes it pay. In
// the block-RAM case the registered read adds a one-cycle bubble the first time a
// new entry is selected to drain, amortized over that entry's requesters.
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

    // Drain: one queued token per cycle. drain_fault selects the wrapper path.
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
    // Registered (block-RAM) read only pays off once the payload store is deep
    // and wide enough; below that it stays a combinational distributed-RAM read.
    // This predicate must be backend-independent: it gates OUT_REG, which adds a
    // read cycle, so keying it off the vendor-specific FORCE_BRAM macro would
    // desync simulation from synthesis. It mirrors the FPGA FORCE_BRAM policy
    // (depth >= 32 and >= 1024 bits) as a fixed threshold instead.
    localparam PAYLOAD_BRAM = (MSHR_SIZE >= 32) && ((MSHR_SIZE * PAYLOAD_W) >= 1024);

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
    wire [MSHR_SIZE-1:0] live = DEDUP_LIVE_EXCLUDES_FAULT ? (valid_r & ~fault_r) : valid_r;

    wire [NUM_REQS-1:0][IDX_W-1:0] probe_slot;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_probe
        wire [MSHR_SIZE-1:0] m;
        for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_m
            assign m[e] = live[e] && (vpn_r[e] == probe_vpn[l]);
        end
        assign probe_match[l] = (| m);
        // Lowest matching index for this lane, formed alongside the match vector
        // so the allocate path can index it directly.
        reg [IDX_W-1:0] ps;
        always @(*) begin
            ps = '0;
            for (int e = MSHR_SIZE-1; e >= 0; --e) begin
                if (m[e]) begin
                    ps = IDX_W'(e);
                end
            end
        end
        assign probe_slot[l] = ps;
    end

    // Allocate reuses the probing lane's dedup result.
    wire             has_match  = probe_match[alloc_lane];
    wire [IDX_W-1:0] match_slot = probe_slot[alloc_lane];

    wire has_free = (| (~valid_r));
    reg [IDX_W-1:0] free_slot;
    always @(*) begin
        free_slot = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (!valid_r[e]) begin
                free_slot = IDX_W'(e);
            end
        end
    end

    wire [IDX_W-1:0] alloc_target = has_match ? match_slot : free_slot;

    // ---------------------------------------------------------------------
    // Per-entry requester queues, pooled into one shared payload RAM. Entry e
    // owns pool slots [e*QDEPTH, (e+1)*QDEPTH); head/tail/count live in FF. At
    // most one push and one pop per cycle, and within an entry head != tail for
    // 0 < count < QDEPTH, so the two ports never collide -> a single VX_dp_ram
    // serves the whole pool. POOL_BRAM sizes it to distributed RAM (async read,
    // cycle-identical) or block RAM (registered read) automatically.
    // ---------------------------------------------------------------------
    `STATIC_ASSERT((QDEPTH >= 2), ("VX_tlb_mshr requires QDEPTH >= 2"))
    `STATIC_ASSERT(((QDEPTH & (QDEPTH-1)) == 0), ("VX_tlb_mshr requires QDEPTH power-of-two"))
    localparam PW         = `CLOG2(QDEPTH);
    localparam POOL_DEPTH = MSHR_SIZE * QDEPTH;
    localparam POOL_AW    = IDX_W + PW;
    localparam POOL_BRAM  = (POOL_DEPTH >= 32) && ((POOL_DEPTH * QDATA_W) >= 1024);

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
    // Allocate accept
    // ---------------------------------------------------------------------
    assign alloc_ready = has_match ? !q_full[match_slot] : has_free;
    wire alloc_fire = alloc_valid && alloc_ready;

    // ---------------------------------------------------------------------
    // Issue
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] issue_pend = valid_r & ~issued_r;
    wire has_issue = (| issue_pend);
    reg [IDX_W-1:0] issue_sel;
    always @(*) begin
        issue_sel = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (issue_pend[e]) begin
                issue_sel = IDX_W'(e);
            end
        end
    end

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

    assign install_valid = fill_fire && !fill_fault;
    assign install_entry = '{
        level: fill_level,
        vpn:   vpn_r[fill_slot],
        ppn:   fill_ppn,
        flags: fill_flags
    };

    assign fault_valid  = fill_fire && fill_fault;
    assign fault_vpn    = vpn_r[fill_slot];
    assign fault_access = access_r[fill_slot];

    // ---------------------------------------------------------------------
    // Drain slot selection
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] drain_pend = valid_r & filled_r;
    wire has_drain = (| drain_pend);
    reg [IDX_W-1:0] drain_sel;
    always @(*) begin
        drain_sel = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (drain_pend[e]) begin
                drain_sel = IDX_W'(e);
            end
        end
    end

    wire drain_ne = !q_empty[drain_sel];

    // ---------------------------------------------------------------------
    // Requester payload pool: one VX_dp_ram shared by all entries, addressed
    // {entry, position}. Push writes the tail slot, drain reads the head slot.
    // ---------------------------------------------------------------------
    wire [POOL_AW-1:0] pool_waddr = {alloc_target, tail_r[alloc_target]};
    wire [POOL_AW-1:0] pool_raddr = {drain_sel, head_r[drain_sel]};
    wire [QDATA_W-1:0] pool_rdata;

    VX_dp_ram #(
        .DATAW    (QDATA_W),
        .SIZE     (POOL_DEPTH),
        .OUT_REG  (POOL_BRAM),
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

    // A registered (block-RAM) read lands one cycle after the address is
    // presented; a shadow of the read address gates the drain until the head
    // token is valid. Async (distributed) reads are always ready. Because the
    // pool address changes on every pop, the registered path drains one token
    // every other cycle; it is dormant until POOL_BRAM (large configs).
    wire pool_ready;
    if (POOL_BRAM) begin : g_pool_bram
        reg [POOL_AW-1:0] praddr_q;
        always @(posedge clk) begin
            if (reset || flush) begin
                praddr_q <= '0;
            end else begin
                praddr_q <= pool_raddr;
            end
        end
        assign pool_ready = (praddr_q == pool_raddr);
    end else begin : g_pool_async
        assign pool_ready = 1'b1;
    end

    // ---------------------------------------------------------------------
    // Fill-result payload store: {ppn, level, flags}, written at fill and read
    // at drain. Single read port -> a VX_dp_ram FORCE_BRAM sizes to distributed
    // RAM (async) or block RAM (registered) automatically.
    // ---------------------------------------------------------------------
    wire [PAYLOAD_W-1:0] payload_wdata = {fill_ppn, fill_level, fill_flags};
    wire [PAYLOAD_W-1:0] payload_rdata;

    VX_dp_ram #(
        .DATAW    (PAYLOAD_W),
        .SIZE     (MSHR_SIZE),
        .OUT_REG  (PAYLOAD_BRAM),
        .RDW_MODE ("W")
    ) payload_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (fill_fire),
        .wren  (1'b1),
        .waddr (fill_slot[IDX_W-1:0]),
        .wdata (payload_wdata),
        .raddr (drain_sel),
        .rdata (payload_rdata)
    );

    assign {drain_ppn, drain_level, drain_flags} = payload_rdata;

    // With a registered read the payload lands one cycle after a new drain slot
    // is presented; a shadow of the read address gates the drain until valid.
    wire payload_ready;
    if (PAYLOAD_BRAM) begin : g_bram_ready
        reg [IDX_W-1:0] raddr_q;
        always @(posedge clk) begin
            if (reset || flush) begin
                raddr_q <= '0;
            end else begin
                raddr_q <= drain_sel;
            end
        end
        assign payload_ready = (raddr_q == drain_sel);
    end else begin : g_async_ready
        assign payload_ready = 1'b1;
    end

    // ---------------------------------------------------------------------
    // Drain
    // ---------------------------------------------------------------------
    assign drain_valid = has_drain && drain_ne && payload_ready && pool_ready;
    assign drain_fault = fault_r[drain_sel];
    assign drain_qdata = pool_rdata;
    wire drain_pop = drain_valid && drain_ready;

    // ---------------------------------------------------------------------
    // Queue push / pop wiring
    // ---------------------------------------------------------------------
    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_qctrl
        assign q_push[e] = alloc_fire && (alloc_target == IDX_W'(e));
        assign q_pop[e]  = drain_pop && (drain_sel == IDX_W'(e));
    end

    // Free the drained entry once its queue empties with no concurrent push. An
    // empty queue is freed only after the reads are ready so a mid-bubble slot
    // switch cannot drop the entry early.
    wire drain_push = alloc_fire && (alloc_target == drain_sel);
    wire drain_free = has_drain && payload_ready && pool_ready
        && ((drain_ne && drain_pop && (q_size[drain_sel] == SIZE_W'(1)) && !drain_push)
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
                filled_r[fill_slot] <= 1'b1;
                fault_r[fill_slot]  <= fill_fault;
            end
            if (drain_free) begin
                valid_r[drain_sel]  <= 1'b0;
                filled_r[drain_sel] <= 1'b0;
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
