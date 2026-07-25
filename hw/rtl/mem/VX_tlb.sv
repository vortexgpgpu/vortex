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
// translation of its own. It holds the fully-associative entry array
// (parallel per-lane read), the non-blocking miss station (park / replay /
// kill), and the fill path from the shared walker over `tlb_bus`. The parent
// `VX_mmu` drives the VPN probes and consumes the raw lookup results (PPN,
// flags) plus the replay/kill streams, doing the VA→PA splice and permission
// checks itself. Splitting the storage out keeps "TLB = lookup" explicit and
// lets the banked variant live entirely here.
module VX_tlb import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NUM_LANES    = DCACHE_NUM_REQS,
    parameter TLB_SIZE     = `VX_CFG_DTLB_SIZE,
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

    // Per-lane combinational lookup (VPN in, raw translation out).
    input  wire [NUM_LANES-1:0][TLB_VPN_WIDTH-1:0]   lookup_vpn,
    output wire [NUM_LANES-1:0]                       lookup_hit,
    output wire [NUM_LANES-1:0][TLB_PPN_WIDTH-1:0]    lookup_ppn,
    output wire [NUM_LANES-1:0][TLB_FLAGS_WIDTH-1:0]  lookup_flags,
    input  wire [NUM_LANES-1:0]                       access_hit,
    output wire [NUM_LANES-1:0]                       mshr_match,

    // Park a miss (payload is opaque; the parent splices on replay).
    input  wire                      park_valid,
    input  wire [TLB_VPN_WIDTH-1:0]  park_vpn,
    input  tlb_access_e              park_access,
    input  wire                      park_amo,
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
    // ---------------------------------------------------------------------
    // Entry array (fully-associative, parallel per-lane read)
    // ---------------------------------------------------------------------
    wire               install_valid;
    tlb_entry_t        install_entry;
    wire               install_evict;

    VX_tlb_l1_cam #(
        .NUM_LANES (NUM_LANES),
        .TLB_SIZE  (TLB_SIZE)
    ) cam (
        .clk           (clk),
        .reset         (reset),
        .lookup_vpn    (lookup_vpn),
        .lookup_hit    (lookup_hit),
        .lookup_ppn    (lookup_ppn),
        .lookup_flags  (lookup_flags),
        `UNUSED_PIN (lookup_ppn_raw),
        `UNUSED_PIN (lookup_level),
        .access_hit    (access_hit),
        .install_valid (install_valid),
        .install_entry (install_entry),
        .install_evict (install_evict),
        .flush         (flush)
    );
`ifndef PERF_ENABLE
    `UNUSED_VAR (install_evict)
`endif

    // ---------------------------------------------------------------------
    // Miss station
    // ---------------------------------------------------------------------
    wire [`UP(ID_WIDTH)-1:0]   tlb_req_id;
    tlb_access_e               tlb_req_access;
    wire                       tlb_req_amo;
    wire [TLB_VPN_WIDTH-1:0]   tlb_req_vpn;

    VX_tlb_l1_mshr #(
        .NUM_LANES    (NUM_LANES),
        .MSHR_SIZE    (MSHR_SIZE),
        .REPLAY_DEPTH (REPLAY_DEPTH),
        .PAYLOAD_W    (PAYLOAD_W),
        .ID_WIDTH     (ID_WIDTH)
    ) mshr (
        .clk           (clk),
        .reset         (reset),
        .probe_vpn     (lookup_vpn),
        .probe_match   (mshr_match),
        .park_valid    (park_valid),
        .park_vpn      (park_vpn),
        .park_access   (park_access),
        .park_amo      (park_amo),
        .park_payload  (park_payload),
        .park_ready    (park_ready),
        .tlb_req_valid (tlb_bus_if.req_valid),
        .tlb_req_id    (tlb_req_id),
        .tlb_req_access(tlb_req_access),
        .tlb_req_amo   (tlb_req_amo),
        .tlb_req_vpn   (tlb_req_vpn),
        .tlb_req_ready (tlb_bus_if.req_ready),
        .tlb_rsp_valid (tlb_bus_if.rsp_valid),
        .tlb_rsp_id    (tlb_bus_if.rsp_data.id),
        .tlb_rsp_fault (tlb_bus_if.rsp_data.fault),
        .tlb_rsp_level (tlb_bus_if.rsp_data.level),
        .tlb_rsp_ppn   (tlb_bus_if.rsp_data.ppn),
        .tlb_rsp_flags (tlb_bus_if.rsp_data.flags),
        .tlb_rsp_ready (tlb_bus_if.rsp_ready),
        .install_valid (install_valid),
        .install_entry (install_entry),
        .replay_valid  (replay_valid),
        .replay_payload(replay_payload),
        .replay_ppn    (replay_ppn),
        .replay_level  (replay_level),
        .replay_flags  (replay_flags),
        .replay_ready  (replay_ready),
        .kill_valid    (kill_valid),
        .kill_ready    (kill_ready),
        .fault_valid   (mshr_fault_valid),
        .fault_vpn     (mshr_fault_vpn),
        .fault_access  (mshr_fault_access),
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
    wire [`CLOG2(NUM_LANES+1)-1:0] n_hits;
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
            if (install_valid && install_evict) begin
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
