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

// L1 TLB miss station: parks missing requests, dedups by VPN so same-page
// misses issue one walk, and replays parked requests when the fill lands.
// The payload is opaque; the parent splices the translated address on
// replay. A valid entry keeps matching new same-VPN requests until its
// replay FIFO drains, so those requests queue behind it in arrival order.
module VX_tlb_l1_mshr import VX_tlb_pkg::*; #(
    parameter NUM_REQS    = 4,
    parameter MSHR_SIZE    = 4,
    parameter REPLAY_DEPTH = 2,
    parameter PAYLOAD_W    = 1,
    parameter ID_WIDTH     = `CLOG2(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

    // Per-lane VPN probe: does the VPN match a live (non-fault) entry?
    input  wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0] probe_vpn,
    output wire [NUM_REQS-1:0]                     probe_match,

    // Park a miss (at most one per cycle).
    input  wire                      park_valid,
    input  wire [TLB_VPN_WIDTH-1:0]  park_vpn,
    input  tlb_access_e              park_access,
    input  wire                      park_amo,
    input  wire [PAYLOAD_W-1:0]      park_payload,
    output wire                      park_ready,

    // TLB miss/fill bus (to the walker complex).
    output wire                       tlb_req_valid,
    output wire [`UP(ID_WIDTH)-1:0]   tlb_req_id,
    output tlb_access_e               tlb_req_access,
    output wire                       tlb_req_amo,
    output wire [TLB_VPN_WIDTH-1:0]   tlb_req_vpn,
    input  wire                       tlb_req_ready,

    input  wire                       tlb_rsp_valid,
    input  wire [`UP(ID_WIDTH)-1:0]   tlb_rsp_id,
    input  wire                       tlb_rsp_fault,
    input  wire [TLB_LEVEL_WIDTH-1:0] tlb_rsp_level,
    input  wire [TLB_PPN_WIDTH-1:0]   tlb_rsp_ppn,
    input  wire [TLB_FLAGS_WIDTH-1:0] tlb_rsp_flags,
    output wire                       tlb_rsp_ready,

    // Install the fill into the CAM (one per cycle, non-fault fills only).
    output wire                       install_valid,
    output tlb_entry_t                install_entry,

    // Replay a parked request into a lane pipeline.
    output wire                       replay_valid,
    output wire [PAYLOAD_W-1:0]       replay_payload,
    output wire [TLB_PPN_WIDTH-1:0]   replay_ppn,
    output wire [TLB_LEVEL_WIDTH-1:0] replay_level,
    output wire [TLB_FLAGS_WIDTH-1:0] replay_flags,
    input  wire                       replay_ready,

    // Kill a parked request whose walk faulted (payload shares replay_payload):
    // the requester tears the access down instead of translating it.
    output wire                       kill_valid,
    input  wire                       kill_ready,

    // First fault sideband (registered upstream).
    output wire                       fault_valid,
    output wire [TLB_VPN_WIDTH-1:0]   fault_vpn,
    output tlb_access_e               fault_access,

    input  wire                       flush,
    output wire                       empty
);
    localparam IDX_W  = `CLOG2(MSHR_SIZE);
    localparam SIZE_W = `CLOG2(REPLAY_DEPTH + 1);

    reg [MSHR_SIZE-1:0]              valid_r;
    reg [MSHR_SIZE-1:0]              issued_r;
    reg [MSHR_SIZE-1:0]              filled_r;
    reg [MSHR_SIZE-1:0]              fault_r;
    reg [TLB_VPN_WIDTH-1:0]         vpn_r    [MSHR_SIZE];
    tlb_access_e                    access_r [MSHR_SIZE];
    reg [MSHR_SIZE-1:0]             amo_r;
    reg [TLB_PPN_WIDTH-1:0]         ppn_r    [MSHR_SIZE];
    reg [TLB_LEVEL_WIDTH-1:0]       level_r  [MSHR_SIZE];
    reg [TLB_FLAGS_WIDTH-1:0]       flags_r  [MSHR_SIZE];

    // ---------------------------------------------------------------------
    // VPN probes and park target selection
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] live = valid_r & ~fault_r;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_probe
        wire [MSHR_SIZE-1:0] m;
        for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_m
            assign m[e] = live[e] && (vpn_r[e] == probe_vpn[l]);
        end
        assign probe_match[l] = (| m);
    end

    wire [MSHR_SIZE-1:0] park_hit;
    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_park_hit
        assign park_hit[e] = live[e] && (vpn_r[e] == park_vpn);
    end
    wire has_match = (| park_hit);

    reg [IDX_W-1:0] match_slot;
    always @(*) begin
        match_slot = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (park_hit[e]) begin
                match_slot = IDX_W'(e);
            end
        end
    end

    wire has_free = (| ~valid_r);
    reg [IDX_W-1:0] free_slot;
    always @(*) begin
        free_slot = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (!valid_r[e]) begin
                free_slot = IDX_W'(e);
            end
        end
    end

    wire [IDX_W-1:0] park_slot = has_match ? match_slot : free_slot;

    // ---------------------------------------------------------------------
    // Replay FIFOs (one per entry)
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0]             fifo_empty;
    wire [MSHR_SIZE-1:0]             fifo_full;
    wire [MSHR_SIZE-1:0][SIZE_W-1:0] fifo_size;
    wire [MSHR_SIZE-1:0][PAYLOAD_W-1:0] fifo_dout;
    wire [MSHR_SIZE-1:0]             fifo_push;
    wire [MSHR_SIZE-1:0]             fifo_pop;

    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_fifo
        VX_fifo_queue #(
            .DATAW  (PAYLOAD_W),
            .DEPTH  (REPLAY_DEPTH),
            .LUTRAM (1)
        ) replay_fifo (
            .clk      (clk),
            .reset    (reset || flush),
            .push     (fifo_push[e]),
            .pop      (fifo_pop[e]),
            .data_in  (park_payload),
            .data_out (fifo_dout[e]),
            .empty    (fifo_empty[e]),
            `UNUSED_PIN (alm_empty),
            .full     (fifo_full[e]),
            `UNUSED_PIN (alm_full),
            .size     (fifo_size[e])
        );
    end

    // ---------------------------------------------------------------------
    // Park accept
    // ---------------------------------------------------------------------
    wire can_append = has_match && !fifo_full[match_slot];
    wire can_alloc  = !has_match && has_free;
    assign park_ready = can_append || can_alloc;
    wire park_fire = park_valid && park_ready;

    // ---------------------------------------------------------------------
    // Issue
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] issue_pend = valid_r & ~issued_r & ~fault_r;
    wire has_issue = (| issue_pend);
    reg [IDX_W-1:0] issue_slot;
    always @(*) begin
        issue_slot = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (issue_pend[e]) begin
                issue_slot = IDX_W'(e);
            end
        end
    end

    assign tlb_req_valid  = has_issue;
    assign tlb_req_id     = `UP(ID_WIDTH)'(issue_slot);
    assign tlb_req_access = access_r[issue_slot];
    assign tlb_req_amo    = amo_r[issue_slot];
    assign tlb_req_vpn    = vpn_r[issue_slot];
    wire issue_fire = tlb_req_valid && tlb_req_ready;

    // ---------------------------------------------------------------------
    // Fill
    // ---------------------------------------------------------------------
    assign tlb_rsp_ready = 1'b1;
    wire fill_fire = tlb_rsp_valid && tlb_rsp_ready;

    assign install_valid = fill_fire && !tlb_rsp_fault;
    assign install_entry = '{
        level: tlb_rsp_level,
        vpn:   vpn_r[tlb_rsp_id],
        ppn:   tlb_rsp_ppn,
        flags: tlb_rsp_flags
    };

    assign fault_valid  = fill_fire && tlb_rsp_fault;
    assign fault_vpn    = vpn_r[tlb_rsp_id];
    assign fault_access = access_r[tlb_rsp_id];

    // ---------------------------------------------------------------------
    // Drain / replay
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] drain_pend = valid_r & filled_r;
    wire has_drain = (| drain_pend);
    reg [IDX_W-1:0] drain_slot;
    always @(*) begin
        drain_slot = '0;
        for (int e = MSHR_SIZE-1; e >= 0; --e) begin
            if (drain_pend[e]) begin
                drain_slot = IDX_W'(e);
            end
        end
    end

    wire drain_fault = fault_r[drain_slot];
    wire drain_ne    = !fifo_empty[drain_slot];

    assign replay_valid   = has_drain && !drain_fault && drain_ne;
    assign replay_payload = fifo_dout[drain_slot];
    assign replay_ppn     = ppn_r[drain_slot];
    assign replay_level   = level_r[drain_slot];
    assign replay_flags   = flags_r[drain_slot];

    // A faulted entry drains its parked requests as kills, one per cycle.
    assign kill_valid = has_drain && drain_fault && drain_ne;

    wire replay_fire = replay_valid && replay_ready;
    wire kill_fire   = kill_valid && kill_ready;
    wire drain_pop   = has_drain && drain_ne && (drain_fault ? kill_fire : replay_fire);

    // ---------------------------------------------------------------------
    // FIFO push / pop wiring
    // ---------------------------------------------------------------------
    for (genvar e = 0; e < MSHR_SIZE; ++e) begin : g_fifo_ctrl
        assign fifo_push[e] = park_fire && (park_slot == IDX_W'(e));
        assign fifo_pop[e]  = drain_pop && (drain_slot == IDX_W'(e));
    end

    // Free the drained entry once its FIFO empties with no concurrent push.
    wire drain_push  = park_fire && (park_slot == drain_slot);
    wire drain_free  = has_drain
        && ((drain_ne && drain_pop && (fifo_size[drain_slot] == SIZE_W'(1)) && !drain_push)
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
            if (park_fire && !has_match) begin
                // Allocate a fresh entry for this VPN.
                valid_r[free_slot]  <= 1'b1;
                issued_r[free_slot] <= 1'b0;
                filled_r[free_slot] <= 1'b0;
                fault_r[free_slot]  <= 1'b0;
                vpn_r[free_slot]    <= park_vpn;
                access_r[free_slot] <= park_access;
                amo_r[free_slot]    <= park_amo;
            end
            if (issue_fire) begin
                issued_r[issue_slot] <= 1'b1;
            end
            if (fill_fire) begin
                filled_r[tlb_rsp_id] <= 1'b1;
                fault_r[tlb_rsp_id]  <= tlb_rsp_fault;
                ppn_r[tlb_rsp_id]    <= tlb_rsp_ppn;
                level_r[tlb_rsp_id]  <= tlb_rsp_level;
                flags_r[tlb_rsp_id]  <= tlb_rsp_flags;
            end
            if (drain_free) begin
                valid_r[drain_slot]  <= 1'b0;
                filled_r[drain_slot] <= 1'b0;
            end
        end
    end

    assign empty = ~(| valid_r);

endmodule
