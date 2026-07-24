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

// Per-client L1 TLB stage. Each lane translates in place (parallel CAM read,
// permission check, PPN/offset splice, one registered output) so N lanes
// translate per cycle. Misses park in a shared miss station and the lane
// keeps going (hit-under-miss); the fill replays the parked requests. A
// bypass request (BARE mode, flush/IO/OM attr) skips translation entirely.
module VX_tlb_l1 import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES    = DCACHE_NUM_REQS,
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

    input wire [`VX_CFG_XLEN-1:0] satp,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    VX_mem_bus_if.slave   core_bus_if [NUM_LANES],
    VX_mem_bus_if.master  mem_bus_if  [NUM_LANES],
    VX_tlb_bus_if.master  tlb_bus_if,
    VX_tlb_flush_if.slave  flush_if,

    output wire                       fault_valid,
    output wire [TLB_VPN_WIDTH-1:0]   fault_vpn,
    output tlb_access_e               fault_access,
    output wire                       empty
);
    `UNUSED_PARAM (INSTANCE_ID)

    localparam DATA_WIDTH    = DATA_SIZE * 8;
    localparam LANE_W        = `UP(`CLOG2(NUM_LANES));
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

    // SATP mode: translation active vs BARE.
    // The L1 only needs the SATP mode bit; the root PPN is consumed by the
    // walker. Sv32 mode = satp[31]; Sv39 mode field = satp[63:60] != 0.
`ifdef VX_VM_ADDR_MODE_SV32
    wire satp_active = satp[31];
    `UNUSED_VAR (satp[30:0])
`else
    wire satp_active = (| satp[`VX_CFG_XLEN-1 -: 4]);
    `UNUSED_VAR (satp[`VX_CFG_XLEN-5:0])
`endif

    function automatic [TLB_VPN_WIDTH-1:0] vpn_mask (input [TLB_LEVEL_WIDTH-1:0] level);
        vpn_mask = {TLB_VPN_WIDTH{1'b1}} << (level * TLB_LEVEL_BITS);
    endfunction

    function automatic tlb_access_e access_of (input logic rw);
        access_of = (EXEC_SIDE != 0) ? TLB_ACC_EX : (rw ? TLB_ACC_WR : TLB_ACC_RD);
    endfunction

    // ---------------------------------------------------------------------
    // Per-lane request decode + CAM lookup
    // ---------------------------------------------------------------------
    wire [NUM_LANES-1:0][FIELDS_W-1:0]      req_fields;
    wire [NUM_LANES-1:0][ADDR_WIDTH-1:0]    req_addr;
    wire [NUM_LANES-1:0][ATTR_WIDTH-1:0]    req_attr;
    wire [NUM_LANES-1:0]                    req_rw;
    wire [NUM_LANES-1:0]                    req_valid;
    wire [NUM_LANES-1:0]                    req_bypass;
    wire [NUM_LANES-1:0][TLB_VPN_WIDTH-1:0] req_vpn;
    tlb_access_e                            req_acc [NUM_LANES];
    wire [NUM_LANES-1:0]                    req_amo;

    wire [NUM_LANES-1:0][TLB_VPN_WIDTH-1:0] cam_vpn;
    wire [NUM_LANES-1:0]                    cam_hit;
    wire [NUM_LANES-1:0][TLB_PPN_WIDTH-1:0] cam_ppn;
    wire [NUM_LANES-1:0][TLB_FLAGS_WIDTH-1:0] cam_flags;
    wire [NUM_LANES-1:0]                    cam_access_hit;
    wire [NUM_LANES-1:0]                    mshr_match;

    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_decode
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
        assign req_bypass[l] = ~satp_active
                            || req_attr[l][MEM_ATTR_FLUSH_OFFS]
                            || req_attr[l][MEM_ATTR_IO_OFFS]
                            || req_attr[l][MEM_ATTR_OM_OFFS];
        assign req_vpn[l] = req_addr[l][PAGE_OFFSET_W +: TLB_VPN_WIDTH];
        assign req_acc[l] = access_of(req_rw[l]);
        assign req_amo[l] = req_attr[l][MEM_ATTR_AMO_OFFS];
        assign cam_vpn[l] = req_vpn[l];
    end

    // Cross-block signals (declared ahead of the instances that drive them).
    wire                       park_valid;
    wire [TLB_VPN_WIDTH-1:0]   park_vpn;
    tlb_access_e               park_access;
    wire                       park_amo;
    wire [PAYLOAD_W-1:0]       park_payload;
    wire                       park_ready;

    wire                       install_valid;
    tlb_entry_t                install_entry;
    wire                       install_evict;

    wire                       replay_valid;
    wire [PAYLOAD_W-1:0]       replay_payload;
    wire [TLB_PPN_WIDTH-1:0]   replay_ppn;
    wire [TLB_LEVEL_WIDTH-1:0] replay_level;
    wire [TLB_FLAGS_WIDTH-1:0] replay_flags;
    wire                       replay_ready;

    wire                       mshr_fault_valid;
    wire [TLB_VPN_WIDTH-1:0]   mshr_fault_vpn;
    tlb_access_e               mshr_fault_access;
    wire                       mshr_empty;

    wire [`UP(ID_WIDTH)-1:0]   tlb_req_id;
    tlb_access_e               tlb_req_access;
    wire                       tlb_req_amo;
    wire [TLB_VPN_WIDTH-1:0]   tlb_req_vpn;

    wire                       flush_clear;

    VX_tlb_l1_cam #(
        .NUM_LANES (NUM_LANES),
        .TLB_SIZE  (TLB_SIZE)
    ) cam (
        .clk           (clk),
        .reset         (reset),
        .lookup_vpn    (cam_vpn),
        .lookup_hit    (cam_hit),
        .lookup_ppn    (cam_ppn),
        .lookup_flags  (cam_flags),
        .access_hit    (cam_access_hit),
        .install_valid (install_valid),
        .install_entry (install_entry),
        .install_evict (install_evict),
        .flush         (flush_clear)
    );
`ifndef PERF_ENABLE
    `UNUSED_VAR (install_evict)
`endif

    // ---------------------------------------------------------------------
    // Miss station
    // ---------------------------------------------------------------------
    VX_tlb_l1_mshr #(
        .NUM_LANES    (NUM_LANES),
        .MSHR_SIZE    (MSHR_SIZE),
        .REPLAY_DEPTH (REPLAY_DEPTH),
        .PAYLOAD_W    (PAYLOAD_W),
        .ID_WIDTH     (ID_WIDTH)
    ) mshr (
        .clk           (clk),
        .reset         (reset),
        .probe_vpn     (req_vpn),
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
        .fault_valid   (mshr_fault_valid),
        .fault_vpn     (mshr_fault_vpn),
        .fault_access  (mshr_fault_access),
        .flush         (flush_clear),
        .empty         (mshr_empty)
    );

    assign tlb_bus_if.req_data = '{ id: tlb_req_id, access: tlb_req_access, amo: tlb_req_amo, vpn: tlb_req_vpn };

    // ---------------------------------------------------------------------
    // Per-lane request category (mutually exclusive, by priority)
    // ---------------------------------------------------------------------
    wire [NUM_LANES-1:0] cat_bypass;
    wire [NUM_LANES-1:0] cat_park;
    wire [NUM_LANES-1:0] cat_hit;
    wire [NUM_LANES-1:0] cat_pfault;
    wire [NUM_LANES-1:0] perm_hit;

    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_cat
        assign perm_hit[l]   = tlb_perm_ok(cam_flags[l], req_acc[l], req_amo[l]);
        assign cat_bypass[l] = req_valid[l] && req_bypass[l];
        assign cat_park[l]   = req_valid[l] && !req_bypass[l] && (mshr_match[l] || !cam_hit[l]);
        assign cat_hit[l]    = req_valid[l] && !req_bypass[l] && !mshr_match[l] && cam_hit[l] && perm_hit[l];
        assign cat_pfault[l] = req_valid[l] && !req_bypass[l] && !mshr_match[l] && cam_hit[l] && !perm_hit[l];
    end

    // Park arbitration: at most one lane parks a miss per cycle (lowest lane).
    wire [NUM_LANES-1:0] park_sel;
    reg [LANE_W-1:0] park_lane;
    always @(*) begin
        park_lane = '0;
        for (int l = NUM_LANES-1; l >= 0; --l) begin
            if (cat_park[l]) begin
                park_lane = LANE_W'(l);
            end
        end
    end
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_park_sel
        assign park_sel[l] = cat_park[l] && (park_lane == LANE_W'(l));
    end

    assign park_valid   = (| cat_park);
    assign park_vpn     = req_vpn[park_lane];
    assign park_access  = req_acc[park_lane];
    assign park_amo     = req_amo[park_lane];
    assign park_payload = {park_lane, req_fields[park_lane]};

    // ---------------------------------------------------------------------
    // Replay decode
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

    wire [FIELDS_W-1:0] replay_out_fields = {
        replay_rw, replay_taddr,
        replay_fields[F_DATA_LO +: DATA_WIDTH],
        replay_fields[F_BYTEEN_LO +: DATA_SIZE],
        replay_rattr,
        replay_fields[F_TAG_LO +: TAG_WIDTH]
    };

    wire [NUM_LANES-1:0] replay_to_lane;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_replay_sel
        assign replay_to_lane[l] = replay_push && (replay_lane == LANE_W'(l));
    end

    // ---------------------------------------------------------------------
    // Per-lane output pipeline (bypass / hit / replay -> registered stage)
    // ---------------------------------------------------------------------
    wire [NUM_LANES-1:0] pipe_ready;

    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_pipe
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

        // Response path wires straight back.
        assign core_bus_if[l].rsp_valid = mem_bus_if[l].rsp_valid;
        assign core_bus_if[l].rsp_data  = mem_bus_if[l].rsp_data;
        assign mem_bus_if[l].rsp_ready  = core_bus_if[l].rsp_ready;
    end

    assign replay_ready = replay_valid && (replay_perm ? pipe_ready[replay_lane] : 1'b1);

    // ---------------------------------------------------------------------
    // Fault sideband to the fault-latch surface. Structural faults come from
    // the walker via the MSHR; permission faults are detected here on hits
    // and replays.
    // ---------------------------------------------------------------------
    wire perm_fault_any = (| cat_pfault) || replay_drop;
    reg [TLB_VPN_WIDTH-1:0] perm_fault_vpn;
    always @(*) begin
        perm_fault_vpn = replay_vpn;
        for (int l = NUM_LANES-1; l >= 0; --l) begin
            if (cat_pfault[l]) begin
                perm_fault_vpn = req_vpn[l];
            end
        end
    end

    assign fault_valid  = mshr_fault_valid || perm_fault_any;
    assign fault_vpn    = mshr_fault_valid ? mshr_fault_vpn : perm_fault_vpn;
    assign fault_access = mshr_fault_valid ? mshr_fault_access
                        : (EXEC_SIDE != 0) ? TLB_ACC_EX : TLB_ACC_WR;

    // ---------------------------------------------------------------------
    // Flush: invalidate the CAM and miss station once outstanding walks drain.
    // ---------------------------------------------------------------------
    assign flush_clear = flush_if.req && mshr_empty;
    assign flush_if.done = flush_if.req && mshr_empty;

    // ---------------------------------------------------------------------
    // Drain status
    // ---------------------------------------------------------------------
    wire [NUM_LANES-1:0] pipe_busy;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_busy
        assign pipe_busy[l] = mem_bus_if[l].req_valid;
    end
    assign empty = ~(| pipe_busy) && mshr_empty;

    // ---------------------------------------------------------------------
    // Performance counters
    // ---------------------------------------------------------------------
`ifdef PERF_ENABLE
    wire [NUM_LANES-1:0] hit_fire;
    wire [NUM_LANES-1:0] miss_fire;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_perf_lane
        assign hit_fire[l]  = cam_access_hit[l];
        assign miss_fire[l] = cat_park[l] && park_sel[l] && park_ready;
    end
    wire [`CLOG2(NUM_LANES+1)-1:0] n_hits;
    wire [`CLOG2(NUM_LANES+1)-1:0] n_miss;
    `POP_COUNT(n_hits, hit_fire);
    `POP_COUNT(n_miss, miss_fire);

    reg [PERF_CTR_BITS-1:0] perf_reads, perf_hits, perf_misses, perf_evicts, perf_walks;
    always @(posedge clk) begin
        if (reset) begin
            perf_reads  <= '0;
            perf_hits   <= '0;
            perf_misses <= '0;
            perf_evicts <= '0;
            perf_walks  <= '0;
        end else begin
            perf_reads  <= perf_reads  + PERF_CTR_BITS'(n_hits) + PERF_CTR_BITS'(n_miss);
            perf_hits   <= perf_hits   + PERF_CTR_BITS'(n_hits);
            perf_misses <= perf_misses + PERF_CTR_BITS'(n_miss);
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
