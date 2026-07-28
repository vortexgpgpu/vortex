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

// Per-client MMU: lookup (delegated to VX_tlb) + address translation. Each
// lane probes the TLB in parallel, and on a hit this module splices the PPN
// with the page offset and checks permissions in place (N lanes translate
// per cycle). Misses park in the TLB's miss station and the lane keeps going
// (hit-under-miss); the fill replays the parked requests, which this module
// re-splices. A bypass request (BARE mode, flush/IO/OM attr) skips
// translation entirely. The MMU owns the tlb_bus to the shared walker and the
// fault sideband; VX_tlb owns the entry array + miss station.
module VX_mmu import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS    = DCACHE_NUM_REQS,
    parameter TLB_SIZE     = `VX_CFG_DTLB_SIZE,
    parameter MSHR_SIZE    = `VX_CFG_L1_TLB_MSHR_SIZE,
    parameter REPLAY_DEPTH = 2,
    parameter EXEC_SIDE    = 0,
    parameter DATA_SIZE    = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH    = DCACHE_TAG_WIDTH_BASE,
    parameter ADDR_WIDTH   = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter ATTR_WIDTH   = MEM_ATTR_WIDTH,
    parameter ID_WIDTH     = `CLOG2(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

    input wire vm_active,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    VX_mem_bus_if.slave   core_bus_if [NUM_REQS],
    VX_mem_bus_if.master  mem_bus_if  [NUM_REQS],
    VX_tlb_bus_if.master  tlb_bus_if,
    VX_tlb_flush_if.slave  flush_if,

    VX_mmu_fault_if.master fault_if,
    output wire            empty
);
    `UNUSED_PARAM (INSTANCE_ID)

    localparam DATA_WIDTH    = DATA_SIZE * 8;
    localparam LANE_W        = `UP(`CLOG2(NUM_REQS));
    localparam PAGE_OFFSET_W = `VX_VM_PAGE_LOG2_SIZE - `CLOG2(DATA_SIZE);

    // Packed request payload (parked on miss, forwarded on hit/bypass).
    localparam F_TAG_LO    = 0;
    localparam F_ATTR_LO   = F_TAG_LO + TAG_WIDTH;
    localparam F_BYTEEN_LO = F_ATTR_LO + ATTR_WIDTH;
    localparam F_DATA_LO   = F_BYTEEN_LO + DATA_SIZE;
    localparam F_ADDR_LO   = F_DATA_LO + DATA_WIDTH;
    localparam F_RW_LO     = F_ADDR_LO + ADDR_WIDTH;
    localparam FIELDS_W    = F_RW_LO + 1;
    localparam PAYLOAD_W   = LANE_W + FIELDS_W;

    function automatic [TLB_VPN_WIDTH-1:0] vpn_mask (input [TLB_LEVEL_WIDTH-1:0] level);
        vpn_mask = {TLB_VPN_WIDTH{1'b1}} << (level * TLB_LEVEL_BITS);
    endfunction

    function automatic tlb_access_e access_of (input logic rw);
        access_of = (EXEC_SIDE != 0) ? TLB_ACC_EX : (rw ? TLB_ACC_WR : TLB_ACC_RD);
    endfunction

    // ---------------------------------------------------------------------
    // Per-lane request decode
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0][FIELDS_W-1:0]      req_fields;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]    req_addr;
    wire [NUM_REQS-1:0][ATTR_WIDTH-1:0]    req_attr;
    wire [NUM_REQS-1:0]                    req_rw;
    wire [NUM_REQS-1:0]                    req_valid;
    wire [NUM_REQS-1:0]                    req_bypass;
    wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0] req_vpn;
    tlb_access_e                            req_acc [NUM_REQS];
    wire [NUM_REQS-1:0]                    req_amo;

    wire [NUM_REQS-1:0][TLB_VPN_WIDTH-1:0]   cam_vpn;
    wire [NUM_REQS-1:0]                      cam_hit;
    wire [NUM_REQS-1:0][TLB_PPN_WIDTH-1:0]   cam_ppn;
    wire [NUM_REQS-1:0][TLB_FLAGS_WIDTH-1:0] cam_flags;
    wire [NUM_REQS-1:0]                      cam_access_hit;
    wire [NUM_REQS-1:0]                      mshr_match;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_decode
        assign req_valid[l] = core_bus_if[l].req_valid;
        assign req_rw[l]    = core_bus_if[l].req_data.rw;
        assign req_addr[l]  = core_bus_if[l].req_data.addr;
        assign req_attr[l]  = core_bus_if[l].req_data.attr[ATTR_WIDTH-1:0];
        assign req_fields[l] = {
            core_bus_if[l].req_data.rw,
            core_bus_if[l].req_data.addr,
            core_bus_if[l].req_data.data,
            core_bus_if[l].req_data.byteen,
            core_bus_if[l].req_data.attr[ATTR_WIDTH-1:0],
            core_bus_if[l].req_data.tag[TAG_WIDTH-1:0]
        };
        assign req_bypass[l] = ~vm_active
                            || req_attr[l][MEM_ATTR_FLUSH_OFFS]
                            || req_attr[l][MEM_ATTR_IO_OFFS]
                            || req_attr[l][MEM_ATTR_OM_OFFS];
        assign req_vpn[l] = req_addr[l][PAGE_OFFSET_W +: TLB_VPN_WIDTH];
        assign req_acc[l] = access_of(req_rw[l]);
        assign req_amo[l] = req_attr[l][MEM_ATTR_AMO_OFFS];
        assign cam_vpn[l] = req_vpn[l];
    end

    // ---------------------------------------------------------------------
    // TLB storage (lookup + miss station)
    // ---------------------------------------------------------------------
    wire                       park_valid;
    wire [TLB_VPN_WIDTH-1:0]   park_vpn;
    tlb_access_e               park_access;
    wire                       park_amo;
    wire [PAYLOAD_W-1:0]       park_payload;
    wire                       park_ready;

    wire                       replay_valid;
    wire [PAYLOAD_W-1:0]       replay_payload;
    wire [TLB_PPN_WIDTH-1:0]   replay_ppn;
    wire [TLB_LEVEL_WIDTH-1:0] replay_level;
    wire [TLB_FLAGS_WIDTH-1:0] replay_flags;
    wire                       replay_ready;

    wire                       kill_valid;
    wire                       kill_ready;

    wire                       mshr_fault_valid;
    wire [TLB_VPN_WIDTH-1:0]   mshr_fault_vpn;
    tlb_access_e               mshr_fault_access;
    wire                       tlb_empty;

    wire                       flush_clear;

    VX_tlb #(
        .NUM_REQS    (NUM_REQS),
        .TLB_SIZE     (TLB_SIZE),
        .MSHR_SIZE    (MSHR_SIZE),
        .REPLAY_DEPTH (REPLAY_DEPTH),
        .PAYLOAD_W    (PAYLOAD_W),
        .ID_WIDTH     (ID_WIDTH)
    ) tlb (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .mmu_perf      (mmu_perf),
    `endif
        .lookup_vpn    (cam_vpn),
        .lookup_hit    (cam_hit),
        .lookup_ppn    (cam_ppn),
        .lookup_flags  (cam_flags),
        .access_hit    (cam_access_hit),
        .mshr_match    (mshr_match),
        .park_valid    (park_valid),
        .park_vpn      (park_vpn),
        .park_access   (park_access),
        .park_amo      (park_amo),
        .park_payload  (park_payload),
        .park_ready    (park_ready),
        .replay_valid  (replay_valid),
        .replay_payload(replay_payload),
        .replay_ppn    (replay_ppn),
        .replay_level  (replay_level),
        .replay_flags  (replay_flags),
        .replay_ready  (replay_ready),
        .kill_valid    (kill_valid),
        .kill_ready    (kill_ready),
        .mshr_fault_valid  (mshr_fault_valid),
        .mshr_fault_vpn    (mshr_fault_vpn),
        .mshr_fault_access (mshr_fault_access),
        .tlb_bus_if    (tlb_bus_if),
        .flush         (flush_clear),
        .empty         (tlb_empty)
    );

    // ---------------------------------------------------------------------
    // Per-lane request category (mutually exclusive, by priority)
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0] cat_bypass;
    wire [NUM_REQS-1:0] cat_park;
    wire [NUM_REQS-1:0] cat_hit;
    wire [NUM_REQS-1:0] cat_pfault;
    wire [NUM_REQS-1:0] perm_hit;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_cat
        assign perm_hit[l]   = tlb_perm_ok(cam_flags[l], req_acc[l], req_amo[l]);
        assign cat_bypass[l] = req_valid[l] && req_bypass[l];
        assign cat_park[l]   = req_valid[l] && !req_bypass[l] && (mshr_match[l] || !cam_hit[l]);
        assign cat_hit[l]    = req_valid[l] && !req_bypass[l] && !mshr_match[l] && cam_hit[l] && perm_hit[l];
        assign cat_pfault[l] = req_valid[l] && !req_bypass[l] && !mshr_match[l] && cam_hit[l] && !perm_hit[l];
    end

    // Park arbitration: at most one lane parks a miss per cycle (lowest lane).
    wire [NUM_REQS-1:0] park_sel;
    reg [LANE_W-1:0] park_lane;
    always @(*) begin
        park_lane = '0;
        for (int l = NUM_REQS-1; l >= 0; --l) begin
            if (cat_park[l]) begin
                park_lane = LANE_W'(l);
            end
        end
    end
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_park_sel
        assign park_sel[l] = cat_park[l] && (park_lane == LANE_W'(l));
    end

    assign park_valid   = (| cat_park);
    assign park_vpn     = req_vpn[park_lane];
    assign park_access  = req_acc[park_lane];
    assign park_amo     = req_amo[park_lane];
    assign park_payload = {park_lane, req_fields[park_lane]};

    // ---------------------------------------------------------------------
    // Replay decode + translation splice
    // ---------------------------------------------------------------------
    wire [LANE_W-1:0]      replay_lane   = replay_payload[FIELDS_W +: LANE_W];
    wire [FIELDS_W-1:0]    replay_fields = replay_payload[FIELDS_W-1:0];
    wire                   replay_rw     = replay_fields[F_RW_LO];
    wire [ADDR_WIDTH-1:0]  replay_raddr  = replay_fields[F_ADDR_LO +: ADDR_WIDTH];
    wire [ATTR_WIDTH-1:0]  replay_rattr  = replay_fields[F_ATTR_LO +: ATTR_WIDTH];

    wire [TLB_VPN_WIDTH-1:0] replay_vpn = replay_raddr[PAGE_OFFSET_W +: TLB_VPN_WIDTH];
    // Translation replaces the page number with the PPN, so any VA bits above
    // the VPN (the Sv39 sign-extension region) do not reach the physical addr.
    if (ADDR_WIDTH > PAGE_OFFSET_W + TLB_VPN_WIDTH) begin : g_replay_va_hi
        `UNUSED_VAR (replay_raddr[ADDR_WIDTH-1 : PAGE_OFFSET_W + TLB_VPN_WIDTH])
    end
    wire [TLB_VPN_WIDTH-1:0] replay_low = replay_vpn & ~vpn_mask(replay_level);
    wire [TLB_PPN_WIDTH-1:0] replay_ppn_sp = replay_ppn | TLB_PPN_WIDTH'(replay_low);
    wire [ADDR_WIDTH-1:0]    replay_taddr = {replay_ppn_sp, replay_raddr[PAGE_OFFSET_W-1:0]};

    tlb_access_e replay_acc;
    assign replay_acc = access_of(replay_rw);
    wire         replay_amo = replay_rattr[MEM_ATTR_AMO_OFFS];
    wire         replay_perm = tlb_perm_ok(replay_flags, replay_acc, replay_amo);

    wire replay_push = replay_valid && replay_perm;   // forwards into a lane
    wire replay_drop = replay_valid && !replay_perm;  // permission fault

    // Kill teardown: a faulting access — a structural fault drained from the
    // miss station (kill_valid) or a permission fault caught re-checking a
    // replay (replay_drop) — still owes the pipeline a response. A data-side
    // load or atomic is answered with zeroed data; a plain store and every
    // fetch retire silently. `kill_*` reuse the replay decode.
    wire                  kill_any     = kill_valid || replay_drop;
    wire [TAG_WIDTH-1:0]  kill_tag     = replay_fields[F_TAG_LO +: TAG_WIDTH];
    wire                  kill_needs_rsp = (EXEC_SIDE == 0) && (!replay_rw || replay_amo);
    wire [NUM_REQS-1:0]  kill_to_lane;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_kill_sel
        assign kill_to_lane[l] = kill_any && kill_needs_rsp && (replay_lane == LANE_W'(l));
    end
    wire [NUM_REQS-1:0]  core_rsp_ready;
    assign kill_ready = kill_needs_rsp ? core_rsp_ready[replay_lane] : 1'b1;

    wire [FIELDS_W-1:0] replay_out_fields = {
        replay_rw, replay_taddr,
        replay_fields[F_DATA_LO +: DATA_WIDTH],
        replay_fields[F_BYTEEN_LO +: DATA_SIZE],
        replay_rattr,
        replay_fields[F_TAG_LO +: TAG_WIDTH]
    };

    wire [NUM_REQS-1:0] replay_to_lane;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_replay_sel
        assign replay_to_lane[l] = replay_push && (replay_lane == LANE_W'(l));
    end

    // ---------------------------------------------------------------------
    // Per-lane output pipeline (bypass / hit / replay -> registered stage)
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0] pipe_ready;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_pipe
        wire [TLB_PPN_WIDTH-1:0] hit_ppn = cam_ppn[l];
        wire [ADDR_WIDTH-1:0]    hit_taddr = {hit_ppn, req_addr[l][PAGE_OFFSET_W-1:0]};

        wire [FIELDS_W-1:0] new_fields = cat_bypass[l]
            ? req_fields[l]
            : { req_rw[l], hit_taddr,
                req_fields[l][F_DATA_LO +: DATA_WIDTH],
                req_fields[l][F_BYTEEN_LO +: DATA_SIZE],
                req_attr[l],
                req_fields[l][F_TAG_LO +: TAG_WIDTH] };

        wire forward_new = (cat_bypass[l] || cat_hit[l]) && !replay_to_lane[l];

        wire                pipe_valid_in = replay_to_lane[l] || forward_new;
        wire [FIELDS_W-1:0] pipe_data_in  = replay_to_lane[l] ? replay_out_fields : new_fields;

        VX_pipe_buffer #(
            .DATAW (FIELDS_W),
            .DEPTH (1)
        ) out_pipe (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (pipe_valid_in),
            .ready_in  (pipe_ready[l]),
            .data_in   (pipe_data_in),
            .data_out  (mem_bus_if[l].req_data),
            .ready_out (mem_bus_if[l].req_ready),
            .valid_out (mem_bus_if[l].req_valid)
        );

        assign cam_access_hit[l] = cat_hit[l] && !replay_to_lane[l] && pipe_ready[l];

        // Request accept back to the client.
        assign core_bus_if[l].req_ready =
              cat_bypass[l] ? (!replay_to_lane[l] && pipe_ready[l])
            : cat_hit[l]    ? (!replay_to_lane[l] && pipe_ready[l])
            : cat_park[l]   ? (park_sel[l] && park_ready)
            : cat_pfault[l] ? 1'b1
            : 1'b0;

        // Response path wires straight back, except a kill teardown injects a
        // zeroed response ahead of the cache reply for its lane.
        assign core_rsp_ready[l] = core_bus_if[l].rsp_ready;
        assign core_bus_if[l].rsp_valid     = kill_to_lane[l] ? 1'b1 : mem_bus_if[l].rsp_valid;
        assign core_bus_if[l].rsp_data.data = kill_to_lane[l] ? '0 : mem_bus_if[l].rsp_data.data;
        assign core_bus_if[l].rsp_data.tag  = kill_to_lane[l] ? kill_tag : mem_bus_if[l].rsp_data.tag;
        assign mem_bus_if[l].rsp_ready      = core_bus_if[l].rsp_ready && ~kill_to_lane[l];
    end

    assign replay_ready = replay_valid && (replay_perm ? pipe_ready[replay_lane] : kill_ready);

    // ---------------------------------------------------------------------
    // Fault sideband to the fault-latch surface. Structural faults come from
    // the walker via the MSHR; permission faults are detected here on hits
    // and replays.
    // ---------------------------------------------------------------------
    wire perm_fault_any = (| cat_pfault) || replay_drop;
    reg [TLB_VPN_WIDTH-1:0] perm_fault_vpn;
    reg                     perm_fault_amo;
    always @(*) begin
        perm_fault_vpn = replay_vpn;
        perm_fault_amo = replay_amo;
        for (int l = NUM_REQS-1; l >= 0; --l) begin
            if (cat_pfault[l]) begin
                perm_fault_vpn = req_vpn[l];
                perm_fault_amo = req_amo[l];
            end
        end
    end

    wire [TLB_VPN_WIDTH-1:0] fault_vpn_sel = mshr_fault_valid ? mshr_fault_vpn : perm_fault_vpn;

    assign fault_if.valid  = mshr_fault_valid || perm_fault_any;
    // faulting page's base virtual address (VPN placed at the page boundary)
    assign fault_if.va     = `VX_CFG_XLEN'(fault_vpn_sel) << `VX_VM_PAGE_LOG2_SIZE;
    assign fault_if.access = mshr_fault_valid ? mshr_fault_access
                           : (EXEC_SIDE != 0) ? TLB_ACC_EX : TLB_ACC_WR;
    // a structural fault leaves the miss station without its AMO intent
    assign fault_if.amo    = mshr_fault_valid ? 1'b0 : perm_fault_amo;

    // ---------------------------------------------------------------------
    // Flush: invalidate the CAM and miss station once outstanding walks drain.
    // ---------------------------------------------------------------------
    assign flush_clear = flush_if.req && tlb_empty;
    assign flush_if.done = flush_if.req && tlb_empty;

    // ---------------------------------------------------------------------
    // Drain status
    // ---------------------------------------------------------------------
    wire [NUM_REQS-1:0] pipe_busy;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_busy
        assign pipe_busy[l] = mem_bus_if[l].req_valid;
    end
    assign empty = ~(| pipe_busy) && tlb_empty;

endmodule
