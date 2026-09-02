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

// L1 TLB storage: the pure lookup + miss-handling core, with no address
// translation of its own. The entry array is banked: NUM_BANKS CAMs selected
// by the low VPN bits, each with a single lookup port, so a lookup costs
// TLB_SIZE comparators in total instead of NUM_REQS x TLB_SIZE. Lanes that
// contend for one bank are serialized (bank_conflict tells the parent to
// hold the losers; lowest lane wins); lanes on different banks proceed in
// the same cycle. Misses go to the shared non-blocking miss station
// (`VX_tlb_mshr`: park / dedup / replay / kill), so a miss never blocks its
// bank and same-VPN requests join the in-flight walk. The fill installs into
// the bank of the walked VPN. The parent `VX_mmu` drives the VPN probes and
// consumes the raw lookup results (PPN, flags) plus the replay/kill streams,
// doing the VA→PA splice and permission checks itself.
module VX_tlb_l1 import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NUM_REQS     = DCACHE_NUM_REQS,
    parameter TLB_SIZE     = `VX_CFG_DTLB_SIZE,
    parameter NUM_BANKS    = `VX_CFG_L1_TLB_NUM_BANKS,
    parameter MSHR_SIZE    = `VX_CFG_L1_TLB_MSHR_SIZE,
    parameter REPLAY_DEPTH = 2,
    parameter PAYLOAD_W    = 1,
    parameter ID_WIDTH     = `CLOG2(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    // Per-lane combinational lookup (VPN in, raw translation out). A lane
    // that loses its bank's port this cycle gets bank_conflict (parent must
    // hold it); hit/miss is only meaningful for lanes with bank_conflict == 0.
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

    // Structural-fault sideband (from the miss station).
    output wire                       mshr_fault_valid,
    output wire [TLB_VPN_WIDTH-1:0]   mshr_fault_vpn,
    output tlb_access_e               mshr_fault_access,

    // Miss/fill fabric to the shared walker complex.
    VX_tlb_bus_if.master  tlb_bus_if,

    input  wire                       flush,
    output wire                       empty
);
    `STATIC_ASSERT(`IS_POW2(NUM_BANKS), ("NUM_BANKS must be a power of 2"))
    `STATIC_ASSERT((TLB_SIZE % NUM_BANKS) == 0, ("NUM_BANKS must divide TLB_SIZE"))

    localparam ENTRIES_PER_BANK = TLB_SIZE / NUM_BANKS;
    localparam BANK_W = `UP(`CLOG2(NUM_BANKS));
    localparam LANE_W = `UP(`CLOG2(NUM_REQS));

    function automatic logic [BANK_W-1:0] bank_of(input logic [BANK_W-1:0] vpn_lo);
        if (NUM_BANKS == 1) bank_of = '0;
        else                bank_of = vpn_lo;
    endfunction

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
    // Entry array: one single-port CAM per bank, granted lane only.
    // ---------------------------------------------------------------------
    wire               install_valid;
    tlb_entry_t        install_entry;
    wire [BANK_W-1:0]  install_bank = bank_of(install_entry.vpn[BANK_W-1:0]);

    wire [NUM_BANKS-1:0]                       bank_install_valid;
    wire [NUM_BANKS-1:0]                       bank_install_evict;
    wire [NUM_BANKS-1:0]                       bank_lookup_hit;
    wire [NUM_BANKS-1:0][TLB_PPN_WIDTH-1:0]    bank_lookup_ppn;
    wire [NUM_BANKS-1:0][TLB_FLAGS_WIDTH-1:0]  bank_lookup_flags;
    wire [NUM_BANKS-1:0][TLB_VPN_WIDTH-1:0]    bank_lookup_vpn;
    wire [NUM_BANKS-1:0]                       bank_access_hit;
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
        assign bank_lookup_vpn[b]    = lookup_vpn[owner];
        assign bank_access_hit[b]    = access_hit[owner] && (lane_bank[owner] == BANK_W'(b));
        assign bank_install_valid[b] = install_valid && (install_bank == BANK_W'(b));

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
    end

    // ---------------------------------------------------------------------
    // Miss station (shared across banks)
    // ---------------------------------------------------------------------
    wire [`UP(ID_WIDTH)-1:0]   tlb_req_id;
    tlb_access_e               tlb_req_access;
    wire                       tlb_req_amo;
    wire [TLB_VPN_WIDTH-1:0]   tlb_req_vpn;

    // L1 drain glue: faulted entries drain as kills, live ones as replays.
    wire                       drain_valid;
    wire [PAYLOAD_W-1:0]       drain_qdata;
    wire                       drain_fault;
    wire [TLB_PPN_WIDTH-1:0]   drain_ppn;
    wire [TLB_LEVEL_WIDTH-1:0] drain_level;
    wire [TLB_FLAGS_WIDTH-1:0] drain_flags;

    assign replay_valid   = drain_valid && !drain_fault;
    assign replay_payload = drain_qdata;
    assign replay_ppn     = drain_ppn;
    assign replay_level   = drain_level;
    assign replay_flags   = drain_flags;
    assign kill_valid     = drain_valid && drain_fault;
    wire drain_ready = drain_fault ? kill_ready : replay_ready;

    VX_tlb_mshr #(
        .NUM_REQS  (NUM_REQS),
        .MSHR_SIZE (MSHR_SIZE),
        .QDATA_W   (PAYLOAD_W),
        .QDEPTH    (REPLAY_DEPTH),
        .DEDUP_LIVE_EXCLUDES_FAULT (1),
        .ID_WIDTH  (ID_WIDTH)
    ) mshr (
        .clk           (clk),
        .reset         (reset),
        .probe_vpn     (lookup_vpn),
        .probe_match   (mshr_match),
        .alloc_valid   (park_valid),
        .alloc_vpn     (park_vpn),
        .alloc_access  (park_access),
        .alloc_amo     (park_amo),
        .alloc_lane    (park_lane),
        .alloc_qdata   (park_payload),
        .alloc_ready   (park_ready),
        .issue_valid   (tlb_bus_if.req_valid),
        .issue_slot    (tlb_req_id),
        .issue_access  (tlb_req_access),
        .issue_amo     (tlb_req_amo),
        .issue_vpn     (tlb_req_vpn),
        .issue_ready   (tlb_bus_if.req_ready),
        .fill_valid    (tlb_bus_if.rsp_valid),
        .fill_slot     (tlb_bus_if.rsp_data.id),
        .fill_fault    (tlb_bus_if.rsp_data.fault),
        .fill_level    (tlb_bus_if.rsp_data.level),
        .fill_ppn      (tlb_bus_if.rsp_data.ppn),
        .fill_flags    (tlb_bus_if.rsp_data.flags),
        .fill_ready    (tlb_bus_if.rsp_ready),
        .install_valid (install_valid),
        .install_entry (install_entry),
        .fault_valid   (mshr_fault_valid),
        .fault_vpn     (mshr_fault_vpn),
        .fault_access  (mshr_fault_access),
        .drain_valid   (drain_valid),
        .drain_qdata   (drain_qdata),
        .drain_fault   (drain_fault),
        .drain_ppn     (drain_ppn),
        .drain_level   (drain_level),
        .drain_flags   (drain_flags),
        .drain_ready   (drain_ready),
        .flush         (flush),
        .empty         (empty)
    );

    assign tlb_bus_if.req_data = '{
        id:     tlb_req_id,
        access: tlb_req_access,
        amo:    tlb_req_amo,
        vpn:    tlb_req_vpn
    };

    // ---------------------------------------------------------------------
    // Performance counters
    // ---------------------------------------------------------------------
`ifdef PERF_ENABLE
    wire [`CLOG2(NUM_REQS+1)-1:0] n_hits;
    `POP_COUNT(n_hits, access_hit);
    wire miss_ev = park_valid && park_ready;

    reg [PERF_CTR_BITS-1:0] perf_reads, perf_hits, perf_misses, perf_evicts, perf_walks;
    always @(posedge clk) begin
        if (reset) begin
            perf_reads  <= '0;
            perf_hits   <= '0;
            perf_misses <= '0;
            perf_evicts <= '0;
            perf_walks  <= '0;
        end else begin
            perf_reads  <= perf_reads  + PERF_CTR_BITS'(n_hits) + PERF_CTR_BITS'(miss_ev);
            perf_hits   <= perf_hits   + PERF_CTR_BITS'(n_hits);
            perf_misses <= perf_misses + PERF_CTR_BITS'(miss_ev);
            if (| (bank_install_valid & bank_install_evict)) begin
                perf_evicts <= perf_evicts + PERF_CTR_BITS'(1);
            end
            if (tlb_bus_if.req_valid && tlb_bus_if.req_ready) begin
                perf_walks <= perf_walks + PERF_CTR_BITS'(1);
            end
        end
    end

    assign mmu_perf.tlb_reads     = perf_reads;
    assign mmu_perf.tlb_hits      = perf_hits;
    assign mmu_perf.tlb_misses    = perf_misses;
    assign mmu_perf.tlb_evictions = perf_evicts;
    assign mmu_perf.ptw_walks     = perf_walks;
    assign mmu_perf.ptw_latency   = '0;
`endif

endmodule
