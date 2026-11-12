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

// Per-client MMU: lookup (delegated to VX_tlb_l1) + address translation. Each
// lane probes the TLB in parallel, and on a hit this module splices the PPN
// with the page offset and checks permissions in place (N lanes translate
// per cycle). Misses park in the TLB's miss station and the lane keeps going
// (hit-under-miss); the fill replays the parked requests, which this module
// re-splices. A bypass request (BARE mode, flush/IO/OM attr) skips
// translation entirely. The MMU owns the tlb_bus to the shared walker and the
// fault sideband; VX_tlb_l1 owns the entry array + miss station.
module VX_mmu import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS    = DCACHE_NUM_REQS,
    parameter TLB_SIZE     = `VX_CFG_DTLB_SIZE,
    parameter MSHR_SIZE    = `VX_CFG_L1_TLB_MSHR_SIZE,
    parameter REPLAY_DEPTH = 2,
    parameter EXEC_SIDE    = 0,
    // 0: the clients only read, so a parked request keeps no write data or
    // byte enables (don't-care on a read).
    parameter WRITE_ENABLE = 1,
    // Response stage per lane toward the client (TO_OUT_BUF encoding). Keeps
    // the injected-response select off the client's response cone.
    parameter RSP_OUT_BUF  = 0,
    // 1: a parking lane is first registered, and allocates in the miss
    // station the next cycle. Takes the lane arbitration off the miss
    // station's update cone; a lane on the staged page holds meanwhile.
    parameter PARK_REG     = 0,
    // 1: each lane takes its requests through a skid register, so the accept,
    // which follows the lookup of the offered request, reaches the client
    // registered. No added latency.
    parameter REQ_SKID     = 0,
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
    // Per lane: nothing of the lane's is in the MMU (front stages, fault slot,
    // and, where flushes are ordered, the miss station).
    output wire [NUM_REQS-1:0] lane_idle,
    output wire            empty
);
    `UNUSED_PARAM (INSTANCE_ID)

    localparam DATA_WIDTH    = DATA_SIZE * 8;

    // The client's bus as the MMU drives it: requests reach the lookup through
    // an optional skid register (REQ_SKID), responses through an optional
    // MMU-owned stage.
    VX_mem_bus_if #(
        .DATA_SIZE  (DATA_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .ATTR_WIDTH (ATTR_WIDTH)
    ) skid_if [NUM_REQS] ();

    VX_mem_bus_if #(
        .DATA_SIZE  (DATA_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .ATTR_WIDTH (ATTR_WIDTH)
    ) req_if [NUM_REQS] ();

    localparam REQ_DATAW = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + `UP(ATTR_WIDTH) + TAG_WIDTH;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_lane_bus
        wire [REQ_DATAW-1:0] skid_data_in = core_bus_if[l].req_data;
        wire [REQ_DATAW-1:0] skid_data_out;

        VX_bypass_buffer #(
            .DATAW    (REQ_DATAW),
            .PASSTHRU (REQ_SKID == 0)
        ) req_skid (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_bus_if[l].req_valid),
            .ready_in  (core_bus_if[l].req_ready),
            .data_in   (skid_data_in),
            .data_out  (skid_data_out),
            .valid_out (skid_if[l].req_valid),
            .ready_out (skid_if[l].req_ready)
        );
        assign skid_if[l].req_data = skid_data_out;

        assign core_bus_if[l].rsp_valid = skid_if[l].rsp_valid;
        assign core_bus_if[l].rsp_data  = skid_if[l].rsp_data;
        assign skid_if[l].rsp_ready     = core_bus_if[l].rsp_ready;

        VX_mem_bus_slice #(
            .DATA_SIZE   (DATA_SIZE),
            .TAG_WIDTH   (TAG_WIDTH),
            .ADDR_WIDTH  (ADDR_WIDTH),
            .ATTR_WIDTH  (ATTR_WIDTH),
            .REQ_OUT_BUF (0),
            .RSP_OUT_BUF (RSP_OUT_BUF)
        ) rsp_buf (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (skid_if[l]),
            .bus_out_if (req_if[l])
        );
    end

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
    // Parked fields: a read-only client's drop {data, byteen}.
    localparam PFIELDS_W   = (WRITE_ENABLE != 0) ? FIELDS_W : (FIELDS_W - DATA_WIDTH - DATA_SIZE);
    localparam PAYLOAD_W   = LANE_W + PFIELDS_W;


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
    wire [NUM_REQS-1:0][$bits(tlb_access_e)-1:0] cam_acc;
    wire [NUM_REQS-1:0]                      cam_perm;
    wire [NUM_REQS-1:0]                      cam_access_hit;
    wire [NUM_REQS-1:0]                      mshr_order;
    wire [NUM_REQS-1:0]                      mshr_join;

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_decode
        assign req_valid[l] = req_if[l].req_valid;
        assign req_rw[l]    = req_if[l].req_data.rw;
        assign req_addr[l]  = req_if[l].req_data.addr;
        assign req_attr[l]  = req_if[l].req_data.attr[ATTR_WIDTH-1:0];
        assign req_fields[l] = {
            req_if[l].req_data.rw,
            req_if[l].req_data.addr,
            req_if[l].req_data.data,
            req_if[l].req_data.byteen,
            req_if[l].req_data.attr[ATTR_WIDTH-1:0],
            req_if[l].req_data.tag[TAG_WIDTH-1:0]
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
    wire [LANE_W-1:0]          park_lane;
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

    VX_tlb_l1 #(
        .NUM_REQS    (NUM_REQS),
        .TLB_SIZE     (TLB_SIZE),
        .MSHR_SIZE    (MSHR_SIZE),
        .REPLAY_DEPTH (REPLAY_DEPTH),
        .PAYLOAD_W    (PAYLOAD_W),
        .ALLOC_PROBE  (PARK_REG),
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
        .lookup_acc    (cam_acc),
        .lookup_amo    (req_amo),
        .lookup_perm   (cam_perm),
        .access_hit    (cam_access_hit),
        .mshr_order    (mshr_order),
        .mshr_join     (mshr_join),
        .park_valid    (park_valid),
        .park_vpn      (park_vpn),
        .park_access   (park_access),
        .park_amo      (park_amo),
        .park_lane     (park_lane),
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
    wire [NUM_REQS-1:0] stage_hold;   // the lane's page is in the park stage
    wire [NUM_REQS-1:0] pf_accept;     // permission fault taken this cycle
    wire [NUM_REQS-1:0] pf_rsp_busy;   // a fault response is owed on the lane
    wire [NUM_REQS-1:0] lane_parked;   // the lane has requests in the miss station

    // The join below reads the per-lane parked count, kept only for writing clients.
    `STATIC_ASSERT ((EXEC_SIDE != 0) || (PARK_REG != 0) || (WRITE_ENABLE != 0),
                    ("invalid parameter: a data-side MMU without a registered park must be WRITE_ENABLE"))

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_cat
        // A flush token holds behind its lane's parked requests (see g_parked).
        wire flush_hold = req_attr[l][MEM_ATTR_FLUSH_OFFS] && lane_parked[l];
        wire xlat = req_valid[l] && !req_bypass[l] && !stage_hold[l];
        // A hit joins its page's live entry only to stay behind the lane's own
        // parked requests (per-lane program order). Joining otherwise would funnel
        // every access to a hot page through the one-per-cycle miss station
        // drain, lowest lane first. A fetch needs no order (one request in flight
        // per warp). A registered park joins always: its stage is a cycle stale.
        assign mshr_order[l] = (EXEC_SIDE == 0) && ((PARK_REG != 0) || lane_parked[l]);
        assign cam_acc[l]    = req_acc[l];
        assign cat_bypass[l] = req_valid[l] && req_bypass[l] && !flush_hold;
        assign cat_park[l]   = xlat && (mshr_join[l] || !cam_hit[l]);
        assign cat_hit[l]    = xlat && !mshr_join[l] && cam_hit[l] && cam_perm[l];
        assign cat_pfault[l] = xlat && !mshr_join[l] && cam_hit[l] && !cam_perm[l];
    end

    // Park arbitration: at most one lane parks a miss per cycle (lowest lane).
    // The parked fields are taken with a one-hot select straight off the lane
    // categories; the encoded lane only rides in the payload. The one-hot comes
    // from a prefix OR (MODEL 2), so a lane's park grant, and with it the lane's
    // accept, depends only on the lanes below it.
    localparam PARK_W = TLB_VPN_WIDTH + $bits(tlb_access_e) + 1 + FIELDS_W;

    wire [NUM_REQS-1:0] park_sel;
    wire                sel_valid;
    wire                sel_ready;
    wire [LANE_W-1:0]   sel_lane;
    VX_priority_encoder #(
        .N     (NUM_REQS),
        .MODEL (2)
    ) park_enc (
        .data_in    (cat_park),
        .onehot_out (park_sel),
        .index_out  (sel_lane),
        .valid_out  (sel_valid)
    );

    wire [NUM_REQS-1:0][PARK_W-1:0] park_src;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_park_src
        assign park_src[l] = {req_vpn[l], req_acc[l], req_amo[l], req_fields[l]};
    end
    // park_sel[l] is cat_park[l] && park_first[l]. A lane's accept reads
    // park_first under its own cat_park, so its translation result enters the
    // accept only once, at the category select.
    wire [NUM_REQS-1:0] park_first;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_park_first
        if (l == 0) begin : g_first
            assign park_first[l] = 1'b1;
        end else begin : g_rest
            assign park_first[l] = ~(| cat_park[l-1:0]);
        end
    end

    wire [PARK_W-1:0] park_out;
    VX_onehot_mux #(
        .DATAW (PARK_W),
        .N     (NUM_REQS)
    ) park_mux (
        .data_in  (park_src),
        .sel_in   (park_sel),
        .data_out (park_out)
    );
    wire [TLB_VPN_WIDTH-1:0] sel_vpn = park_out[PARK_W-1 -: TLB_VPN_WIDTH];

    // ---------------------------------------------------------------------
    // Replay decode + translation splice
    // ---------------------------------------------------------------------
    wire [LANE_W-1:0]      replay_lane    = replay_payload[PFIELDS_W +: LANE_W];
    wire [PFIELDS_W-1:0]   replay_pfields = replay_payload[PFIELDS_W-1:0];
    wire [PFIELDS_W-1:0]   park_pfields;
    wire [FIELDS_W-1:0]    replay_fields;

    if (WRITE_ENABLE != 0) begin : g_park_wr
        assign park_pfields  = park_out[FIELDS_W-1:0];
        assign replay_fields = replay_pfields;
    end else begin : g_park_ro
        // A read's data and byte enables are don't-care below the L1; the
        // replay presents it as a full-line read.
        assign park_pfields  = {park_out[FIELDS_W-1:F_ADDR_LO], park_out[F_BYTEEN_LO-1:0]};
        assign replay_fields = {replay_pfields[PFIELDS_W-1:F_BYTEEN_LO], DATA_WIDTH'(0), {DATA_SIZE{1'b1}},
                                replay_pfields[F_BYTEEN_LO-1:0]};
        `UNUSED_VAR (park_out[F_ADDR_LO-1:F_BYTEEN_LO])
        `RUNTIME_ASSERT (~(sel_valid && sel_ready) || ~park_out[F_RW_LO],
                         ("%t: *** %s read-only MMU parked a write, addr=0x%0h", $time, INSTANCE_ID,
                          park_out[F_ADDR_LO +: ADDR_WIDTH]))
    end

    // The selected lane's park, straight into the miss station or through a
    // one-entry stage. The stage accepts only while empty, so its ready is a
    // flop; it allocates the next cycle against its own VPN (ALLOC_PROBE).
    localparam STAGE_W = TLB_VPN_WIDTH + $bits(tlb_access_e) + 1 + LANE_W + PFIELDS_W;
    wire [STAGE_W-1:0] sel_park = {sel_vpn, park_out[FIELDS_W+1 +: $bits(tlb_access_e)],
                                   park_out[FIELDS_W], sel_lane, park_pfields};
    wire [STAGE_W-1:0] stage_park;
    wire stage_valid;

    if (PARK_REG != 0) begin : g_park_reg
        reg              stage_valid_r;
        reg [STAGE_W-1:0] stage_r;
        always @(posedge clk) begin
            if (reset) begin
                stage_valid_r <= 1'b0;
            end else if (sel_valid && sel_ready) begin
                stage_valid_r <= 1'b1;
            end else if (park_ready) begin
                stage_valid_r <= 1'b0;
            end
        end
        always @(posedge clk) begin
            if (sel_ready) begin
                stage_r <= sel_park;
            end
        end
        assign sel_ready   = ~stage_valid_r;
        assign stage_valid = stage_valid_r;
        assign stage_park  = stage_r;
        // Until it allocates, a later request on the staged page holds: it would
        // miss the miss-station probe and could overtake the staged request.
        wire [TLB_VPN_WIDTH-1:0] stage_vpn = stage_r[STAGE_W-1 -: TLB_VPN_WIDTH];
        for (genvar l = 0; l < NUM_REQS; ++l) begin : g_hold
            assign stage_hold[l] = stage_valid_r && (req_vpn[l] == stage_vpn);
        end
    end else begin : g_park_direct
        assign sel_ready   = park_ready;
        assign stage_valid = sel_valid;
        assign stage_park  = sel_park;
        assign stage_hold  = '0;
    end

    assign park_valid   = stage_valid;
    assign park_vpn     = stage_park[STAGE_W-1 -: TLB_VPN_WIDTH];
    assign park_access  = tlb_access_e'(stage_park[LANE_W + PFIELDS_W + 1 +: $bits(tlb_access_e)]);
    assign park_amo     = stage_park[LANE_W + PFIELDS_W];
    assign park_lane    = stage_park[PFIELDS_W +: LANE_W];
    assign park_payload = stage_park[LANE_W + PFIELDS_W - 1 : 0];
    wire                   replay_rw     = replay_fields[F_RW_LO];
    wire [ADDR_WIDTH-1:0]  replay_raddr  = replay_fields[F_ADDR_LO +: ADDR_WIDTH];
    wire [ATTR_WIDTH-1:0]  replay_rattr  = replay_fields[F_ATTR_LO +: ATTR_WIDTH];

    wire [TLB_VPN_WIDTH-1:0] replay_vpn = replay_raddr[PAGE_OFFSET_W +: TLB_VPN_WIDTH];
    // Translation replaces the page number with the PPN, so any VA bits above
    // the VPN (the Sv39 sign-extension region) do not reach the physical addr.
    if (ADDR_WIDTH > PAGE_OFFSET_W + TLB_VPN_WIDTH) begin : g_replay_va_hi
        `UNUSED_VAR (replay_raddr[ADDR_WIDTH-1 : PAGE_OFFSET_W + TLB_VPN_WIDTH])
    end
    wire [TLB_VPN_WIDTH-1:0] replay_low = replay_vpn & ~tlb_vpn_mask(replay_level);
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

        // SIZE=2/OUT_REG=1 registers this stage's outgoing request onto
        // mem_bus_if and presents a registered ready_in (pipe_ready), keeping the
        // accept cone shallow; a DEPTH=1 pipe would leave ready_in combinational.
        VX_elastic_buffer #(
            .DATAW   (FIELDS_W),
            .SIZE    (2),
            .OUT_REG (1)
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

        // A permission fault on a hit never reaches memory, but a data-side
        // load or atomic still owes its requester a response, as a kill does:
        // it waits in a per-lane slot and is answered with zeroed data.
        wire                 pf_rsp_valid;
        wire [TAG_WIDTH-1:0] pf_rsp_tag;
        if (EXEC_SIDE == 0) begin : g_pf_rsp
            wire pf_needs_rsp = !req_rw[l] || req_amo[l];
            reg                 pf_rsp_valid_r;
            reg [TAG_WIDTH-1:0] pf_rsp_tag_r;
            wire pf_rsp_fire = pf_rsp_valid_r && !kill_to_lane[l] && req_if[l].rsp_ready;
            // An empty slot loads unconditionally so the tag's enable stays off
            // the CAM lookup cone; only the valid bit depends on the fault.
            always @(posedge clk) begin
                if (reset) begin
                    pf_rsp_valid_r <= 1'b0;
                end else if (pf_rsp_valid_r) begin
                    pf_rsp_valid_r <= ~pf_rsp_fire;
                end else begin
                    pf_rsp_valid_r <= cat_pfault[l] && pf_needs_rsp;
                end
            end
            always @(posedge clk) begin
                if (~pf_rsp_valid_r) begin
                    pf_rsp_tag_r <= req_fields[l][F_TAG_LO +: TAG_WIDTH];
                end
            end
            assign pf_accept[l] = cat_pfault[l] && !(pf_needs_rsp && pf_rsp_valid_r);
            assign pf_rsp_valid = pf_rsp_valid_r;
            assign pf_rsp_tag   = pf_rsp_tag_r;
        end else begin : g_pf_no_rsp
            // A fetch is answered by the fault report alone.
            assign pf_accept[l] = cat_pfault[l];
            assign pf_rsp_valid = 1'b0;
            assign pf_rsp_tag   = '0;
        end
        assign pf_rsp_busy[l] = pf_rsp_valid;

        // Request accept back to the client.
        assign req_if[l].req_ready =
              cat_bypass[l] ? (!replay_to_lane[l] && pipe_ready[l])
            : cat_hit[l]    ? (!replay_to_lane[l] && pipe_ready[l])
            : cat_park[l]   ? (park_first[l] && sel_ready)
            : cat_pfault[l] ? pf_accept[l]
            : 1'b0;

        // Response path wires straight back, except a kill teardown or a
        // permission fault injects a zeroed response ahead of the cache reply.
        // The injected tag is chosen ahead of the one select on the reply.
        wire                 inj_rsp = kill_to_lane[l] || pf_rsp_valid;
        wire [TAG_WIDTH-1:0] inj_tag = kill_to_lane[l] ? kill_tag : pf_rsp_tag;
        assign core_rsp_ready[l] = req_if[l].rsp_ready;
        assign req_if[l].rsp_valid     = inj_rsp ? 1'b1 : mem_bus_if[l].rsp_valid;
        assign req_if[l].rsp_data.data = inj_rsp ? '0 : mem_bus_if[l].rsp_data.data;
        assign req_if[l].rsp_data.tag  = inj_rsp ? inj_tag : mem_bus_if[l].rsp_data.tag;
        assign mem_bus_if[l].rsp_ready = req_if[l].rsp_ready && ~inj_rsp;
    end

    assign replay_ready = replay_valid && (replay_perm ? pipe_ready[replay_lane] : kill_ready);

    // ---------------------------------------------------------------------
    // Per-lane parked count. A flush token bypasses translation, so it would
    // overtake its lane's parked requests; it holds until they have left the
    // miss station instead, reaching the cache behind them as a lane's
    // requests always do (a FLUSH_FORWARD cache forwards it behind that
    // port's writes). Only a data-side MMU with writing clients needs it: a
    // read overtaken by a flush token is still answered correctly.
    // ---------------------------------------------------------------------
    localparam PCNT_W = `CLOG2(MSHR_SIZE * REPLAY_DEPTH + 2);
    // A park leaves its lane when the stage (or the miss station) takes it.
    wire              park_take      = sel_valid && sel_ready;
    wire [LANE_W-1:0] park_take_lane = sel_lane;
    wire              drain_take     = replay_ready || (kill_valid && kill_ready);

    if ((WRITE_ENABLE != 0) && (EXEC_SIDE == 0)) begin : g_flush_order
        for (genvar l = 0; l < NUM_REQS; ++l) begin : g_parked
            reg [PCNT_W-1:0] parked_cnt_r;
            wire inc = park_take && (park_take_lane == LANE_W'(l));
            wire dec = drain_take && (replay_lane == LANE_W'(l));
            always @(posedge clk) begin
                if (reset) begin
                    parked_cnt_r <= '0;
                end else if (inc && !dec) begin
                    parked_cnt_r <= parked_cnt_r + PCNT_W'(1);
                end else if (dec && !inc) begin
                    parked_cnt_r <= parked_cnt_r - PCNT_W'(1);
                end
            end
            `RUNTIME_ASSERT (reset || ~(dec && !inc) || (parked_cnt_r != '0),
                             ("%t: *** %s lane %0d drained more than it parked", $time, INSTANCE_ID, l))
            assign lane_parked[l] = (parked_cnt_r != '0);
        end
    end else begin : g_no_flush_order
        assign lane_parked = '0;
        `UNUSED_VAR (park_take)
        `UNUSED_VAR (park_take_lane)
        `UNUSED_VAR (drain_take)
    end

    // ---------------------------------------------------------------------
    // Fault sideband to the fault-latch surface. Structural faults come from
    // the walker via the MSHR; permission faults are detected here on hits
    // and replays.
    // ---------------------------------------------------------------------
    wire perm_fault_any = (| pf_accept) || replay_drop;
    reg [TLB_VPN_WIDTH-1:0] perm_fault_vpn;
    reg                     perm_fault_amo;
    always @(*) begin
        perm_fault_vpn = replay_vpn;
        perm_fault_amo = replay_amo;
        for (int l = NUM_REQS-1; l >= 0; --l) begin
            if (pf_accept[l]) begin
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
    // The MMU is drained only when no request is anywhere in its front pipeline
    // (incoming, skid or output stage) and the TLB miss station is idle.
    // Omitting the incoming stage would let `empty` assert a cycle early while
    // a request is still in flight, releasing the socket drain gate prematurely.
    wire [NUM_REQS-1:0] pipe_busy;
    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_busy
        assign pipe_busy[l] = req_if[l].req_valid        // incoming or skid
                           || mem_bus_if[l].req_valid    // output stage
                           || pf_rsp_busy[l];            // owed fault response
        assign lane_idle[l] = ~pipe_busy[l] && ~lane_parked[l];
    end
    assign empty = ~(| pipe_busy) && tlb_empty && ~(PARK_REG != 0 && stage_valid);

endmodule
