// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// Banked L1 TLB storage + miss station behind the VX_mmu parent contract.
// The entry array is split into NUM_BANKS single-ported banks selected by
// the low VPN bits: each bank answers at most one lane per cycle
// (bank_conflict tells the parent to hold the other lanes), and each bank
// holds at most one parked miss — a miss blocks its bank for the walk's
// duration while the other banks keep hitting. A same-VPN request waits on
// its bank's in-flight walk (mshr_match) instead of joining a queue.
// Trades a full multi-port CAM + shared MSHR for per-bank ports and slots:
// cheaper lookup hardware at scale, one outstanding walk per bank.
module VX_tlb_l1 import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NUM_REQS   = DCACHE_NUM_REQS,
    parameter TLB_SIZE   = `VX_CFG_DTLB_SIZE,
    parameter NUM_BANKS  = 4,
    parameter PAYLOAD_W  = 1,
    parameter ID_WIDTH   = `CLOG2(NUM_BANKS)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    // Per-lane lookup. A lane that loses its bank's port this cycle gets
    // bank_conflict (parent must hold it); hit/miss is only meaningful for
    // lanes with bank_conflict == 0.
    input  wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0]   lookup_vpn,
    input  wire [NUM_REQS-1:0]                       lookup_valid,
    output wire [NUM_REQS-1:0]                       lookup_hit,
    output wire [NUM_REQS-1:0][TLB_PPN_WIDTH-1:0]    lookup_ppn,
    output wire [NUM_REQS-1:0][TLB_FLAGS_WIDTH-1:0]  lookup_flags,
    output wire [NUM_REQS-1:0]                       bank_conflict,
    input  wire [NUM_REQS-1:0]                       access_hit,
    output wire [NUM_REQS-1:0]                       mshr_match,

    // Park a miss (payload is opaque; the parent splices on replay).
    input  wire                      park_valid,
    input  wire [TLB_VPN_WIDTH-1:0]  park_vpn,
    input  tlb_access_e              park_access,
    input  wire                      park_amo,
    input  wire [`UP(`CLOG2(NUM_REQS))-1:0] park_lane,
    input  wire [PAYLOAD_W-1:0]      park_payload,
    output wire                      park_ready,

    // Replay a parked request once its fill lands.
    output wire                       replay_valid,
    output wire [PAYLOAD_W-1:0]       replay_payload,
    output wire [TLB_PPN_WIDTH-1:0]   replay_ppn,
    output wire [TLB_LEVEL_WIDTH-1:0] replay_level,
    output wire [TLB_FLAGS_WIDTH-1:0] replay_flags,
    input  wire                       replay_ready,

    // Kill a parked request whose walk faulted.
    output wire                       kill_valid,
    input  wire                       kill_ready,

    // Structural-fault sideband.
    output wire                       mshr_fault_valid,
    output wire [TLB_VPN_WIDTH-1:0]   mshr_fault_vpn,
    output tlb_access_e               mshr_fault_access,

    // Miss/fill fabric to the shared walker complex (id = bank index).
    VX_tlb_bus_if.master  tlb_bus_if,

    input  wire                       flush,
    output wire                       empty
);
    `STATIC_ASSERT(`IS_POW2(NUM_BANKS), ("NUM_BANKS must be a power of 2"))
    `STATIC_ASSERT((TLB_SIZE % NUM_BANKS) == 0, ("NUM_BANKS must divide TLB_SIZE"))
    `STATIC_ASSERT(ID_WIDTH >= `CLOG2(NUM_BANKS), ("bank index must fit the bus id"))

    localparam ENTRIES_PER_BANK = TLB_SIZE / NUM_BANKS;
    localparam BANK_W = `UP(`CLOG2(NUM_BANKS));
    localparam LANE_W = `UP(`CLOG2(NUM_REQS));

    function automatic logic [BANK_W-1:0] bank_of(input logic [BANK_W-1:0] vpn_lo);
        if (NUM_BANKS == 1) bank_of = '0;
        else                bank_of = vpn_lo;
    endfunction

    // ---------------------------------------------------------------------
    // Per-bank parked-miss slot
    // ---------------------------------------------------------------------
    typedef enum logic [1:0] {
        B_IDLE, B_WALK_REQ, B_WALK_WAIT, B_DRAIN
    } bank_state_e;

    bank_state_e                bk_state   [NUM_BANKS];
    logic [TLB_VPN_WIDTH-1:0]   bk_vpn     [NUM_BANKS];
    tlb_access_e                bk_access  [NUM_BANKS];
    logic                       bk_amo     [NUM_BANKS];
    logic [PAYLOAD_W-1:0]       bk_payload [NUM_BANKS];
    logic                       bk_fault   [NUM_BANKS];
    logic [TLB_PPN_WIDTH-1:0]   bk_ppn     [NUM_BANKS];
    logic [TLB_LEVEL_WIDTH-1:0] bk_level   [NUM_BANKS];
    logic [TLB_FLAGS_WIDTH-1:0] bk_flags   [NUM_BANKS];
    // A walk in flight when the flush arrived resolved against the old page
    // table: drop its fill and re-walk (see the shared walker's discipline).
    logic                       bk_stale   [NUM_BANKS];

    // ---------------------------------------------------------------------
    // Bank lookup port arbitration: lowest contending lane wins the bank.
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0][BANK_W-1:0] lane_bank;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_lane_bank
        assign lane_bank[l] = bank_of(lookup_vpn[l][BANK_W-1:0]);
    end

    wire [NUM_REQS-1:0] lane_grant;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_grant
        logic older_same_bank;
        always @(*) begin
            older_same_bank = 1'b0;
            for (int k = 0; k < l; ++k) begin
                if (lookup_valid[k] && (lane_bank[k] == lane_bank[l])) begin
                    older_same_bank = 1'b1;
                end
            end
        end
        assign lane_grant[l]    = lookup_valid[l] && !older_same_bank;
        assign bank_conflict[l] = lookup_valid[l] && older_same_bank;
    end

    // ---------------------------------------------------------------------
    // Entry storage: one CAM per bank, granted lane only.
    // ---------------------------------------------------------------------
    wire [NUM_BANKS-1:0]                       bank_install_valid;
    tlb_entry_t                                install_entry;
    wire [NUM_BANKS-1:0]                       bank_lookup_hit;
    wire [NUM_BANKS-1:0][TLB_PPN_WIDTH-1:0]    bank_lookup_ppn;
    wire [NUM_BANKS-1:0][TLB_FLAGS_WIDTH-1:0]  bank_lookup_flags;
    wire [NUM_BANKS-1:0][TLB_VPN_WIDTH-1:0]    bank_lookup_vpn;
    wire [NUM_BANKS-1:0]                       bank_access_hit;
    wire [NUM_BANKS-1:0]                       bank_install_evict;
`ifndef PERF_ENABLE
    `UNUSED_VAR (bank_install_evict)
`endif

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_banks
        logic [LANE_W-1:0] owner;
        always @(*) begin
            owner = '0;
            for (int l = NUM_REQS-1; l >= 0; --l) begin
                if (lane_grant[l] && (lane_bank[l] == BANK_W'(b))) begin
                    owner = LANE_W'(l);
                end
            end
        end
        assign bank_lookup_vpn[b] = lookup_vpn[owner];
        assign bank_access_hit[b] = access_hit[owner] && (lane_bank[owner] == BANK_W'(b));

        VX_tlb_cam #(
            .NUM_REQS (1),
            .TLB_SIZE (ENTRIES_PER_BANK)
        ) cam (
            .clk           (clk),
            .reset         (reset),
            .lookup_vpn    (bank_lookup_vpn[b]),
            .lookup_hit    (bank_lookup_hit[b]),
            .lookup_ppn    (bank_lookup_ppn[b]),
            .lookup_flags  (bank_lookup_flags[b]),
            `UNUSED_PIN (lookup_ppn_raw),
            `UNUSED_PIN (lookup_level),
            .access_hit    (bank_access_hit[b]),
            .install_valid (bank_install_valid[b]),
            .install_entry (install_entry),
            .install_evict (bank_install_evict[b]),
            .flush         (flush)
        );
    end

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_lane_out
        assign lookup_hit[l]   = lane_grant[l] && bank_lookup_hit[lane_bank[l]];
        assign lookup_ppn[l]   = bank_lookup_ppn[lane_bank[l]];
        assign lookup_flags[l] = bank_lookup_flags[lane_bank[l]];
        // A lane whose VPN matches its bank's in-flight walk waits for that
        // fill rather than re-walking (categorized cat_park; park_ready holds
        // it off until the bank drains and the entry installs).
        assign mshr_match[l]   = lane_grant[l]
                              && (bk_state[lane_bank[l]] != B_IDLE)
                              && (bk_vpn[lane_bank[l]] == lookup_vpn[l]);
    end

    // ---------------------------------------------------------------------
    // Park: one slot per bank; the bank must be idle.
    // ---------------------------------------------------------------------
    wire [BANK_W-1:0] park_bank = bank_of(park_vpn[BANK_W-1:0]);
    assign park_ready = !flush && (bk_state[park_bank] == B_IDLE);
    wire park_fire = park_valid && park_ready;
    `UNUSED_VAR (park_lane)

    // ---------------------------------------------------------------------
    // Walk issue: round-robin over banks in B_WALK_REQ.
    // ---------------------------------------------------------------------
    reg  [BANK_W-1:0] issue_rr;
    logic [BANK_W-1:0] issue_sel;
    logic              issue_any;
    always @(*) begin
        issue_sel = '0;
        issue_any = 1'b0;
        for (int i = NUM_BANKS-1; i >= 0; --i) begin
            automatic logic [BANK_W-1:0] b = BANK_W'(int'(issue_rr) + i + 1);
            if (bk_state[b] == B_WALK_REQ) begin
                issue_sel = b;
                issue_any = 1'b1;
            end
        end
    end

    assign tlb_bus_if.req_valid = issue_any;
    assign tlb_bus_if.req_data = '{
        id:     `UP(ID_WIDTH)'(issue_sel),
        access: bk_access[issue_sel],
        amo:    bk_amo[issue_sel],
        vpn:    bk_vpn[issue_sel]
    };
    wire issue_fire = tlb_bus_if.req_valid && tlb_bus_if.req_ready;

    // ---------------------------------------------------------------------
    // Fill: install (unless stale/faulted) and stage the bank for drain.
    // ---------------------------------------------------------------------
    assign tlb_bus_if.rsp_ready = 1'b1;
    wire fill_fire = tlb_bus_if.rsp_valid && tlb_bus_if.rsp_ready;
    wire [BANK_W-1:0] fill_bank = BANK_W'(tlb_bus_if.rsp_data.id);
    wire fill_ok = fill_fire && !tlb_bus_if.rsp_data.fault
                && !bk_stale[fill_bank] && !flush;

    assign install_entry = '{
        level: tlb_bus_if.rsp_data.level,
        vpn:   bk_vpn[fill_bank],
        ppn:   tlb_bus_if.rsp_data.ppn,
        flags: tlb_bus_if.rsp_data.flags
    };
    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_install
        assign bank_install_valid[b] = fill_ok && (fill_bank == BANK_W'(b));
    end

    // ---------------------------------------------------------------------
    // Drain: one bank per cycle, round-robin; faulted walks kill.
    // ---------------------------------------------------------------------
    reg  [BANK_W-1:0] drain_rr;
    logic [BANK_W-1:0] drain_sel;
    logic              drain_any;
    always @(*) begin
        drain_sel = '0;
        drain_any = 1'b0;
        for (int i = NUM_BANKS-1; i >= 0; --i) begin
            automatic logic [BANK_W-1:0] b = BANK_W'(int'(drain_rr) + i + 1);
            if (bk_state[b] == B_DRAIN) begin
                drain_sel = b;
                drain_any = 1'b1;
            end
        end
    end

    assign replay_valid   = drain_any && !bk_fault[drain_sel];
    assign replay_payload = bk_payload[drain_sel];
    assign replay_ppn     = bk_ppn[drain_sel];
    assign replay_level   = bk_level[drain_sel];
    assign replay_flags   = bk_flags[drain_sel];
    assign kill_valid     = drain_any && bk_fault[drain_sel];
    wire drain_fire = drain_any && (bk_fault[drain_sel] ? kill_ready : replay_ready);

    // Structural faults surface as the kill drains.
    assign mshr_fault_valid  = kill_valid && kill_ready;
    assign mshr_fault_vpn    = bk_vpn[drain_sel];
    assign mshr_fault_access = bk_access[drain_sel];

    // ---------------------------------------------------------------------
    // Bank state
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                bk_state[b] <= B_IDLE;
                bk_stale[b] <= 1'b0;
            end
            issue_rr <= '0;
            drain_rr <= '0;
        end else begin
            if (park_fire) begin
                bk_state  [park_bank] <= B_WALK_REQ;
                bk_vpn    [park_bank] <= park_vpn;
                bk_access [park_bank] <= park_access;
                bk_amo    [park_bank] <= park_amo;
                bk_payload[park_bank] <= park_payload;
                bk_stale  [park_bank] <= 1'b0;
            end
            if (issue_fire) begin
                bk_state[issue_sel] <= B_WALK_WAIT;
                issue_rr <= issue_sel;
            end
            if (fill_fire) begin
                if (bk_stale[fill_bank]) begin
                    // stale walk: discard the result and walk again
                    bk_state[fill_bank] <= B_WALK_REQ;
                    bk_stale[fill_bank] <= 1'b0;
                end else begin
                    bk_state[fill_bank] <= B_DRAIN;
                    bk_fault[fill_bank] <= tlb_bus_if.rsp_data.fault;
                    bk_ppn  [fill_bank] <= tlb_bus_if.rsp_data.ppn;
                    bk_level[fill_bank] <= tlb_bus_if.rsp_data.level;
                    bk_flags[fill_bank] <= tlb_bus_if.rsp_data.flags;
                end
            end
            if (drain_fire) begin
                bk_state[drain_sel] <= B_IDLE;
                drain_rr <= drain_sel;
            end
            if (flush) begin
                for (int b = 0; b < NUM_BANKS; ++b) begin
                    if (bk_state[b] == B_WALK_WAIT) begin
                        bk_stale[b] <= 1'b1;
                    end
                end
            end
        end
    end

    logic any_busy;
    always @(*) begin
        any_busy = 1'b0;
        for (int b = 0; b < NUM_BANKS; ++b) begin
            if (bk_state[b] != B_IDLE) any_busy = 1'b1;
        end
    end
    assign empty = !any_busy;

    // ---------------------------------------------------------------------
    // Performance counters
    // ---------------------------------------------------------------------
`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_reads, perf_hits, perf_misses, perf_evicts;
    logic [`CLOG2(NUM_REQS+1)-1:0] reads_now, hits_now;
    always @(*) begin
        reads_now = '0;
        hits_now  = '0;
        for (int l = 0; l < NUM_REQS; ++l) begin
            if (lane_grant[l])                reads_now = reads_now + 1;
            if (lookup_hit[l] && access_hit[l]) hits_now = hits_now + 1;
        end
    end
    always @(posedge clk) begin
        if (reset) begin
            perf_reads  <= '0;
            perf_hits   <= '0;
            perf_misses <= '0;
            perf_evicts <= '0;
        end else begin
            perf_reads  <= perf_reads  + PERF_CTR_BITS'(reads_now);
            perf_hits   <= perf_hits   + PERF_CTR_BITS'(hits_now);
            perf_misses <= perf_misses + PERF_CTR_BITS'(park_fire);
            perf_evicts <= perf_evicts + PERF_CTR_BITS'((| (bank_install_valid & bank_install_evict)));
        end
    end
    assign mmu_perf.tlb_reads     = perf_reads;
    assign mmu_perf.tlb_hits      = perf_hits;
    assign mmu_perf.tlb_misses    = perf_misses;
    assign mmu_perf.tlb_evictions = perf_evicts;
    // Every parked miss issues exactly one walk (re-walks after a flush are
    // counted again on issue).
    assign mmu_perf.ptw_walks     = perf_misses;
    assign mmu_perf.ptw_latency   = '0;
`endif

endmodule
