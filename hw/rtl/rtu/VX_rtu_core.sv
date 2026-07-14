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

// VX_rtu_core — socket-shared ray-traversal engine, and the sole WRITER of the
// hit window (see VX_rtu_bus_if). It is Vulkan-Sim's warp buffer + ray buffer
// (Saed et al., MICRO'22, §III-C-2): rays stage on arrival, a pool of SLOTS
// traverses several of them at once, and the shared front end inside the
// scheduler switches between resident slots whenever one parks on memory.
//
// Two structures, and confusing them is what made every earlier revision of this
// core a one-trace machine with a queue bolted on:
//
//   STAGING  one entry per {src, wid} — per warp of every core this RTU serves.
//            An arm and its RAY beats land here, and they land UNCONDITIONALLY:
//            a warp holds one trace, so its entry is free by construction. That
//            is what makes the arm's ready a constant 1, and therefore what stops
//            a TRACE burst from ever stalling in the in-order SFU while holding
//            the issue lock — the deadlock the issue stage used to carry an RTU
//            trace gate to dodge. It carries nothing about the RTU now, and it
//            never will again: the hazard is gone, not guarded.
//
//   SLOTS    RTU_NUM_SLOTS traversals in flight. A slot OWNS NUM_LANES contexts
//            (NUM_CTX = NUM_SLOTS * NUM_LANES), so its warp's rays walk the BVH
//            CONCURRENTLY with another slot's — and when one slot's contexts are
//            all parked on a node fetch, the scheduler's selector simply finds
//            another slot's. That is the latency hiding; residency alone buys
//            nothing. A slot frees when all its contexts terminate.
//
// Staging is what decouples the two: a ray may arrive with every slot busy, and
// it waits HERE, inside the RTU, instead of back-pressuring the SFU. Nothing the
// core does can stall a warp except the warp's own WAIT.
//
// A traversal that finds a non-opaque hit writes the candidate back and parks at
// its slot's yield barrier. The warp reads the candidate with GETW, runs its
// any-hit / intersection shader, and resumes the walk with CONTINUE; the actions
// arrive on the `req` channel — together with the shader's own t and hitAttribute,
// which is why the walk never has to read anything back — and it finishes into a
// terminal record. The status slot is always the LAST write of a response, because
// writing it is what completes the warp's parked WAIT.

`include "VX_define.vh"

module VX_rtu_core import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter NUM_SRCS  = 1,   // cores this RTU serves
    parameter TAG_WIDTH = 1,
    parameter CACHE_DATA_SIZE = `VX_CFG_MEM_BLOCK_SIZE,
    parameter CACHE_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // window bus (this core is the master)
    VX_rtu_bus_if.slave  rtu_bus_if,

    // RTCache port
    VX_mem_bus_if.master cache_bus_if
);
    // The slot spans the walk writes must each be contiguous, or the base+index
    // addressing below silently targets the wrong slots.
    `STATIC_ASSERT((`VX_RT_OBJECT_RAY_ORIGIN == RTU_RES_BASE + RTU_RES_HIT),
        ("the object ray must abut the hit attributes"))
    `STATIC_ASSERT((`VX_RT_CB_HANDLE == RTU_RES_BASE + RTU_RES_CAND - 1),
        ("the candidate result slots must be one contiguous span"))

    // Register the outgoing bus/cache interfaces at this module boundary so the
    // SLR-crossing seams launch/capture at flops (see VX_rtu_bus_slice). Only
    // the channels this core sources are buffered here; the window registers
    // the ones it sources.
    localparam SRC_WIDTH = `UP(`CLOG2(NUM_SRCS));

    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH),
        .SRC_WIDTH (SRC_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH),
        .SRC_WIDTH   (SRC_WIDTH),
        .ARM_OUT_BUF (0),  // arm_ready is a constant 1; nothing to register
        .REQ_OUT_BUF (0),  // req already registered upstream (unit/arb)
        .SLV_OUT_BUF (3)   // register our outgoing window accesses
    ) rtu_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (rtu_bus_if),
        .bus_out_if (rtu_bus_w)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (CACHE_DATA_SIZE),
        .TAG_WIDTH (CACHE_TAG_WIDTH)
    ) cache_bus_w ();

    VX_mem_bus_slice #(
        .DATA_SIZE   (CACHE_DATA_SIZE),
        .TAG_WIDTH   (CACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3),  // register our outgoing RTCache request
        .RSP_OUT_BUF (0)   // response registered by the RTCache output
    ) cache_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (cache_bus_w),
        .bus_out_if (cache_bus_if)
    );
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8;

    // ── the slot pool ─────────────────────────────────────────────────────
    // A slot owns NUM_LANES contexts. This is the load-bearing identity of the
    // whole design: it is what makes two resident warps TRAVERSE at once rather
    // than merely queue, and it is the only reason a second slot is worth its area.
    localparam NUM_SLOTS = `VX_CFG_RTU_NUM_SLOTS;
    localparam NUM_CTX   = NUM_SLOTS * NUM_LANES;
    localparam SLOT_W    = `LOG2UP(NUM_SLOTS);
    localparam CTX_TAG_W = `LOG2UP(NUM_CTX);

    `STATIC_ASSERT((`VX_CFG_RTU_NUM_CTX == NUM_CTX),
        ("VX_CFG_RTU_NUM_CTX must equal RTU_NUM_SLOTS * NUM_THREADS: a slot owns its contexts"))

    // ── knobs this core does not implement ────────────────────────────────
    // MERGE_DEPTH — the MSHR file that merges duplicate node fetches. This core
    //   issues one request per context per fetch (per-context tags), which is
    //   exactly what MERGE_DEPTH=0 selects. SimX honours the knob, so a config
    //   that raised it would run one machine in simulation and a different one in
    //   hardware: fail the build instead of diverging silently.
    `STATIC_ASSERT((`VX_CFG_RTU_MERGE_DEPTH == 0),
        ("VX_CFG_RTU_MERGE_DEPTH > 0 is not implemented in RTL (SimX-only); the RTL core does not merge node fetches"))

    // ── ray staging: one entry per {src, wid}, its words in BRAM ──────────
    // The two halves of a trace arrive on SEPARATE channels (the arm's scalars, the
    // beats' per-lane words) through separate register slices, so either may land
    // first. Track them independently; an entry is launchable once both are in.
    localparam NUM_STG   = NUM_SRCS * `VX_CFG_NUM_WARPS;
    localparam STG_IDX_W = `LOG2UP(NUM_STG);
    localparam RAY_IDX_W = `CLOG2(RTU_RAY_BEATS);

    reg [NUM_STG-1:0]                             stg_armed;   // the arm's scalars are here
    reg [NUM_STG-1:0]                             stg_full;    // all RTU_RAY_BEATS words are here
    reg [NUM_STG-1:0]                             stg_bound;   // a slot took it AT THE ARM (see below)
    reg [NUM_STG-1:0][SLOT_W-1:0]                 stg_slot;    // ... and this is the slot
    reg [NUM_STG-1:0][RAY_IDX_W-1:0]              stg_beat;    // ray beats landed so far
    reg [NUM_STG-1:0][NUM_LANES-1:0]              stg_mask;
    reg [NUM_STG-1:0][TAG_WIDTH-1:0]              stg_tag;
    reg [NUM_STG-1:0][31:0]                       stg_payload;
    reg [NUM_STG-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] stg_scene;
    reg [NUM_STG-1:0][15:0]                       stg_flags;
    reg [NUM_STG-1:0][15:0]                       stg_cull;

    // ── per-slot traversal state ──────────────────────────────────────────
    localparam [2:0] T_IDLE   = 3'd0,  // free
                     T_FILL   = 3'd1,  // bound at the arm; its ray beats land straight in
                     T_BUSY   = 3'd2,  // traversing
                     T_WRITE  = 3'd3,  // writing the record back (terminal | candidate)
                     T_CBWAIT = 3'd4,  // candidate returned; await the CONTINUE's t
                     T_CBATTR = 3'd5,  // ... and its hitAttribute (CONT beat 1)
                     T_RESUME = 3'd6;  // release this slot's yield barrier
    reg [NUM_SLOTS-1:0][2:0]           tstate;

    reg [NUM_SLOTS-1:0][NUM_LANES-1:0] req_mask;
    rtu_ray_t [NUM_SLOTS-1:0][NUM_LANES-1:0] req_rays;
    reg [NUM_SLOTS-1:0][TAG_WIDTH-1:0] req_tag;
    reg [NUM_SLOTS-1:0][STG_IDX_W-1:0] req_stg;      // the staging entry it launched from
    reg [NUM_SLOTS-1:0][NW_WIDTH-1:0]  req_wid;      // its warp, for the window write
    reg [NUM_SLOTS-1:0][31:0]          req_payload;  // warp-uniform; staged for the callbacks

    // ── the hit record is NOT stored here ─────────────────────────────────
    // The record is read LIVE off the scheduler as the walk writes it, rather than
    // latched here — latching it would be a second copy of state the scheduler already
    // holds in hit_* / yld_*, at ~1 kb of flops per slot.
    //
    // This is sound because the scheduler holds the record still for exactly the
    // window this core needs it:
    //   candidate — yield[s] stays asserted until we answer with resume[s], and the
    //               record is written before that (T_WRITE precedes T_CBWAIT).
    //   terminal  — done[s] pulses, running[s] drops, and hit_* then persist
    //               untouched until the slot is started again, which cannot happen
    //               until this record has been written.
    // Do not "optimise" this back into a latch. The copy WAS the cost.
    wire [CTX_TAG_W-1:0] wctx [NUM_LANES];   // the writing slot's contexts

    // The hitAttribute of the ACCEPTED candidate — the one record field the scheduler
    // does NOT hold (it never sees the shader's attribute). Bound at the verdict: a
    // later IGNORE/DONE carries no attribute, and latching every verdict's rs3 would
    // let it overwrite the accepted one.
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0][31:0] res_hit_attr;

    reg [NUM_SLOTS-1:0]                                         is_cand;
    // the warp's CONTINUE, latched off the req channel: the per-lane action, and
    // the intersection shader's own t and hitAttribute.
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0]  cont_action;
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0][31:0]                    cont_hit_t;
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0][31:0]                    cont_attr;

    // record-write walk counter
    reg [NUM_SLOTS-1:0][RTU_IDX_BITS-1:0] wr_idx;

    reg [NUM_SLOTS-1:0] sch_start_r;

    // A slot whose ray is already loaded but whose PREVIOUS record is still going
    // out. See the load engine: this is what hides the load behind the record write.
    reg [NUM_SLOTS-1:0]                slot_preloaded;
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0] nxt_mask;
    reg [NUM_SLOTS-1:0][TAG_WIDTH-1:0] nxt_tag;
    reg [NUM_SLOTS-1:0][STG_IDX_W-1:0] nxt_stg;
    reg [NUM_SLOTS-1:0][NW_WIDTH-1:0]  nxt_wid;
    reg [NUM_SLOTS-1:0][31:0]          nxt_payload;

    // ── the scheduler: NUM_CTX contexts, NUM_SLOTS concurrent traversals ──
    wire [NUM_CTX-1:0]       sch_mask;
    rtu_ray_t [NUM_CTX-1:0]  sch_rays;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_sch_in
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane
            assign sch_mask[s * NUM_LANES + i] = req_mask[s][i];
            assign sch_rays[s * NUM_LANES + i] = req_rays[s][i];
        end
    end

    wire [NUM_SLOTS-1:0]      sch_busy, sch_done;
    wire [NUM_CTX-1:0]        sch_hit;
    wire [NUM_CTX-1:0][31:0]  sch_t, sch_u, sch_v, sch_prim, sch_geom, sch_inst, sch_custom;
    `UNUSED_VAR (sch_busy)
    // scheduler callback yield barrier
    wire [NUM_SLOTS-1:0]                       sch_yield, sch_resume;
    wire [NUM_CTX-1:0]                         sch_ymask;
    wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]   sch_ycbtype;
    wire [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]    sch_ysbt;
    wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] sch_action;
    wire [NUM_CTX-1:0][31:0]                   sch_action_hit_t;

    // scheduler <-> mem (tagged by context id)
    wire                              m_req_valid, m_req_ready, m_rsp_valid, m_rsp_ready;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] m_req_addr;
    wire [CTX_TAG_W-1:0]              m_req_tag, m_rsp_tag;
    wire [LINE_BITS-1:0]              m_rsp_data;

    // Compile-time walker selection (true-hardware model): RTU_BVH_WIDTH==0
    // builds a flat triangle-list walker; 4/6 build the CW-BVH walker. Both
    // present the same scheduler interface.
    if (RTU_BVH_WIDTH == 0) begin : g_flat_scheduler
        VX_rtu_flat_scheduler #(
            .INSTANCE_ID (INSTANCE_ID),
            .NUM_SLOTS   (NUM_SLOTS),
            .NUM_CTX     (NUM_CTX)
        ) scheduler (
            .clk          (clk),
            .reset        (reset),
            .start        (sch_start_r),
            .mask         (sch_mask),
            .rays         (sch_rays),
            .busy         (sch_busy),
            .done         (sch_done),
            .res_hit      (sch_hit),
            .res_t        (sch_t),
            .res_u        (sch_u),
            .res_v        (sch_v),
            .res_prim     (sch_prim),
            .res_geom     (sch_geom),
            .res_inst     (sch_inst),
            .res_custom   (sch_custom),
            .yield        (sch_yield),
            .yield_mask   (sch_ymask),
            .yield_cbtype (sch_ycbtype),
            .yield_sbt    (sch_ysbt),
            .resume       (sch_resume),
            .action       (sch_action),
            .action_hit_t (sch_action_hit_t),
            .mem_req_valid (m_req_valid),
            .mem_req_addr  (m_req_addr),
            .mem_req_tag   (m_req_tag),
            .mem_req_ready (m_req_ready),
            .mem_rsp_valid (m_rsp_valid),
            .mem_rsp_data  (m_rsp_data),
            .mem_rsp_tag   (m_rsp_tag),
            .mem_rsp_ready (m_rsp_ready)
        );
    end else begin : g_bvh_scheduler
        VX_rtu_bvh_scheduler #(
            .INSTANCE_ID (INSTANCE_ID),
            .NUM_SLOTS   (NUM_SLOTS),
            .NUM_CTX     (NUM_CTX)
        ) scheduler (
            .clk          (clk),
            .reset        (reset),
            .start        (sch_start_r),
            .mask         (sch_mask),
            .rays         (sch_rays),
            .busy         (sch_busy),
            .done         (sch_done),
            .res_hit      (sch_hit),
            .res_t        (sch_t),
            .res_u        (sch_u),
            .res_v        (sch_v),
            .res_prim     (sch_prim),
            .res_geom     (sch_geom),
            .res_inst     (sch_inst),
            .res_custom   (sch_custom),
            .yield        (sch_yield),
            .yield_mask   (sch_ymask),
            .yield_cbtype (sch_ycbtype),
            .yield_sbt    (sch_ysbt),
            .resume       (sch_resume),
            .action       (sch_action),
            .action_hit_t (sch_action_hit_t),
            .mem_req_valid (m_req_valid),
            .mem_req_addr  (m_req_addr),
            .mem_req_tag   (m_req_tag),
            .mem_req_ready (m_req_ready),
            .mem_rsp_valid (m_rsp_valid),
            .mem_rsp_data  (m_rsp_data),
            .mem_rsp_tag   (m_rsp_tag),
            .mem_rsp_ready (m_rsp_ready)
        );
    end

    // ── RTCache port: node/leaf fetch ────────────────────────────────────
    // A CW-BVH4 node is exactly one cache line, so a fetch is one aligned line read
    // tagged with the requesting context id, which is what distinguishes responses.
    // Outstanding requests are bounded by the cache's own MSHRs.
    localparam RTU_LINE_SIZE  = `VX_CFG_MEM_BLOCK_SIZE;
    localparam RTU_LINE_ADDRW = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(RTU_LINE_SIZE);
    `UNUSED_VAR (m_req_addr[`CLOG2(RTU_LINE_SIZE)-1:0])

    `STATIC_ASSERT(CTX_TAG_W <= $bits(cache_bus_w.req_data.tag.value),
        ("rtu fetch tag (%0d bits) does not fit the rtcache tag field", CTX_TAG_W))

    assign cache_bus_w.req_valid        = m_req_valid;
    assign cache_bus_w.req_data.rw      = 1'b0;
    assign cache_bus_w.req_data.addr    = m_req_addr[`VX_CFG_MEM_ADDR_WIDTH-1 -: RTU_LINE_ADDRW];
    assign cache_bus_w.req_data.data    = '0;
    assign cache_bus_w.req_data.byteen  = {RTU_LINE_SIZE{1'b1}};
    assign cache_bus_w.req_data.tag.uuid  = '0;
    assign cache_bus_w.req_data.tag.value = $bits(cache_bus_w.req_data.tag.value)'(m_req_tag);
    assign cache_bus_w.req_data.attr    = '0;
    assign m_req_ready = cache_bus_w.req_ready;

    assign m_rsp_valid = cache_bus_w.rsp_valid;
    assign m_rsp_data  = cache_bus_w.rsp_data.data;
    assign m_rsp_tag   = CTX_TAG_W'(cache_bus_w.rsp_data.tag.value);
    assign cache_bus_w.rsp_ready = m_rsp_ready;
    `UNUSED_VAR (cache_bus_w.rsp_data.tag.uuid)

    // ── the req channel: RAY beats, and the CONTINUE ──────────────────────
    // Always ready: everything arriving here is something this core is already
    // waiting for, so it can never back-pressure (which is what lets a parked
    // traversal stay reachable while another warp's ray streams in).
    assign rtu_bus_w.req_ready = 1'b1;
    wire req_fire = rtu_bus_w.req_valid;
    wire is_ray   = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_RAY);
    wire is_cont  = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_CONT);

    // Every beat names its owner, so it lands in that owner's staging entry no
    // matter which slots are traversing, and several warps' bursts — from several
    // cores — may interleave here safely.
    wire [STG_IDX_W-1:0] req_stg_idx = STG_IDX_W'({rtu_bus_w.req_data.src, rtu_bus_w.req_data.wid});

    // ── the arm channel: ALWAYS ready ─────────────────────────────────────
    // A warp owns a staging entry and can hold only one trace, so an arm always has
    // somewhere to land. This assignment is the deadlock fix: a TRACE burst can
    // never stall in the SFU, so it can never wedge the issue lock.
    assign rtu_bus_w.arm_ready = 1'b1;
    wire arm_fire = rtu_bus_w.arm_valid;
    wire [STG_IDX_W-1:0] arm_stg_idx = STG_IDX_W'({rtu_bus_w.arm_data.src, rtu_bus_w.arm_data.wid});

    // The ABI: a warp WAITs for its trace before arming another. Nothing enforced
    // it, and a violation would overwrite a live entry — corruption, not a hang.
    `RUNTIME_ASSERT(~(arm_fire && stg_armed[arm_stg_idx]),
        ("%t: *** %s: staging entry %0d armed a second trace without waiting for the first",
            $time, INSTANCE_ID, arm_stg_idx))

    // A beat arriving for an entry whose ray is already whole has nowhere to go:
    // that is a warp arming twice, or a lost beat, and it would corrupt silently.
    `RUNTIME_ASSERT(~(is_ray && stg_full[req_stg_idx]),
        ("%t: *** %s: RAY beat for staging entry %0d whose ray is already complete",
            $time, INSTANCE_ID, req_stg_idx))

    // ── the staging RAM: NUM_STG x RTU_RAY_BEATS per-lane words ───────────
    // The bulk state of the staging structure, and the reason it is nearly free.
    // Written a beat at a time as a TRACE burst streams; read back once, by the
    // load engine, into the launching slot's ray registers.
    wire                       ray_ram_rd;
    wire [STG_IDX_W-1:0]       ld_stg;
    wire [RAY_IDX_W-1:0]       ld_idx;
    wire [NUM_LANES-1:0][31:0] ray_ram_q;

    VX_dp_ram #(
        .DATAW    (NUM_LANES * 32),
        .SIZE     (NUM_STG * RTU_RAY_BEATS),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) ray_ram (
        .clk   (clk),
        .reset (reset),
        .read  (ray_ram_rd),
        .write (is_ray && ~ray_bound),   // bound rays never touch the RAM
        .wren  (1'b1),
        .waddr ({req_stg_idx, stg_beat[req_stg_idx]}),
        .wdata (rtu_bus_w.req_data.data),
        .raddr ({ld_stg, ld_idx}),
        .rdata (ray_ram_q)
    );

    // ── the load engine ───────────────────────────────────────────────────
    // One staging entry at a time streams into one free slot. It runs CONCURRENTLY
    // with every traversal: loading is not a state of the traversal FSM, so a slot
    // taking its ray costs the traversing slots nothing.
    localparam [0:0] L_IDLE = 1'd0, L_STREAM = 1'd1;
    reg              lstate;
    reg [SLOT_W-1:0]    ld_slot;
    reg [STG_IDX_W-1:0] ld_stg_r;
    reg [RAY_IDX_W-1:0] ld_issue;
    reg [RAY_IDX_W-1:0] ld_cap_idx;
    reg                 ld_cap_vld;

    // A staging entry is launchable once BOTH halves are in — and once no slot has
    // already claimed it (stg_armed drops at the terminal record, so the claim is
    // tracked by stg_full clearing at the load).
    wire [NUM_STG-1:0] stg_ready = stg_armed & stg_full;

    wire [STG_IDX_W-1:0] pick_stg;
    wire                 pick_stg_valid;
    wire [NUM_STG-1:0]   pick_stg_1h;
    VX_priority_encoder #(
        .N (NUM_STG)
    ) stg_picker (
        .data_in    (stg_ready),
        .onehot_out (pick_stg_1h),
        .index_out  (pick_stg),
        .valid_out  (pick_stg_valid)
    );
    `UNUSED_VAR (pick_stg_1h)

    // A slot the load engine is streaming into is neither free nor loadable again.
    wire [NUM_SLOTS-1:0] slot_filling;
    wire [NUM_SLOTS-1:0] slot_free;
    wire [NUM_SLOTS-1:0] slot_loadable;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_slot_sel
        assign slot_filling[s] = (lstate == L_STREAM) && (ld_slot == SLOT_W'(s));
        assign slot_free[s]    = (tstate[s] == T_IDLE)
                              && ~slot_preloaded[s] && ~slot_filling[s];
        // ── PRELOAD ─────────────────────────────────────────────────────────
        // A slot writing out a TERMINAL record may take its next ray NOW, while that
        // record is still streaming. Nothing observes req_rays during a terminal
        // record: its span is the 7 hit attributes, then payload / hitAttribute /
        // status, and each of those three overrides the word this select produces.
        // (A CANDIDATE record does publish the object ray from req_rays — hence the
        // is_cand guard.) This is what hides the ray load behind the record write, and
        // it is why a one-slot RTU costs nothing versus the old blocking-arm core.
        assign slot_loadable[s] = slot_free[s]
                               || ((tstate[s] == T_WRITE) && ~is_cand[s]
                                   && ~slot_preloaded[s] && ~slot_filling[s]);
    end

    wire [SLOT_W-1:0]    pick_slot;
    wire                 pick_slot_valid;
    wire [NUM_SLOTS-1:0] pick_slot_1h;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) slot_picker (
        .data_in    (slot_free),
        .onehot_out (pick_slot_1h),
        .index_out  (pick_slot),
        .valid_out  (pick_slot_valid)
    );
    `UNUSED_VAR (pick_slot_1h)

    wire [SLOT_W-1:0]    ld_pick;
    wire                 ld_pick_valid;
    wire [NUM_SLOTS-1:0] ld_pick_1h;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) ld_picker (
        .data_in    (slot_loadable),
        .onehot_out (ld_pick_1h),
        .index_out  (ld_pick),
        .valid_out  (ld_pick_valid)
    );
    `UNUSED_VAR (ld_pick_1h)

    // ── the arm-time bind: the zero-latency path, and the common one ──────
    // If a slot is free when the arm lands, that slot takes the trace THERE AND
    // THEN and the ray beats behind the arm stream straight into its traversal
    // registers. Nothing round-trips through the staging RAM, so an uncontended
    // trace starts the cycle its last beat arrives — which is what the old
    // blocking-arm core did, and the ~10 cycles per trace the RAM path costs is
    // exactly what made a correct slot pool look like a regression.
    //
    // The staging RAM is now the OVERFLOW path: it is what a ray uses when every
    // slot is busy. That is the only case it is needed for, and it is precisely
    // the case that used to block the SFU.
    //
    // Bind only if no beat has landed yet: a beat may overtake its own arm (they
    // ride separate channels through separate buffers), and half a ray in the RAM
    // plus half in a slot is a corrupt ray. A late arm simply takes the RAM path.
    wire bind_now = arm_fire && pick_slot_valid
                 && (stg_beat[arm_stg_idx] == RAY_IDX_W'(0))
                 && ~stg_full[arm_stg_idx];

    // The arm wins the free slot; the load engine takes one only if no arm did.
    wire ld_launch = (lstate == L_IDLE) && pick_stg_valid && ld_pick_valid && ~bind_now;

    // A beat belongs to a bound slot if its entry was bound earlier, or is being
    // bound by an arm in THIS cycle (arm and beat 0 can land together).
    wire              ray_bound = stg_bound[req_stg_idx]
                               || (bind_now && (req_stg_idx == arm_stg_idx));
    wire [SLOT_W-1:0] ray_slot  = stg_bound[req_stg_idx] ? stg_slot[req_stg_idx] : pick_slot;

    // ld_stg_r, NOT pick_stg: the picker is combinational over stg_ready, so another
    // warp's ray completing mid-load would re-point it and the load would finish
    // reading somebody else's ray. ld_stg_r latched the choice.
    assign ld_stg     = ld_launch ? pick_stg : ld_stg_r;
    // The launch cycle issues the read for beat 0; ld_issue is not yet pointing at
    // it (it is a register, and it still holds the last load's final index).
    assign ld_idx     = ld_launch ? RAY_IDX_W'(0) : ld_issue;
    assign ray_ram_rd = ld_launch || (lstate == L_STREAM);

    // ── window writes (the `win` channel) ─────────────────────────────────
    // One record, written as an ordered stream: the hit attributes (plus, for a
    // candidate, the object ray and callback metadata), then the warp-uniform
    // payload pointer the callback shaders read, then the accepted hit's
    // hitAttribute, and the status LAST.
    //
    // Slots contend for this channel, and the grant is STICKY for a whole record:
    // the status write is what completes the warp's parked WAIT, so it must not be
    // possible for another slot's beats to land between a record's attributes and
    // its status.
    wire [NUM_SLOTS-1:0] want_win;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_want_win
        assign want_win[s] = (tstate[s] == T_WRITE);
    end

    reg              win_lock;
    reg [SLOT_W-1:0] win_owner_r;

    wire [SLOT_W-1:0]    win_grant;
    wire                 win_grant_valid;
    wire [NUM_SLOTS-1:0] win_grant_1h;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) win_picker (
        .data_in    (want_win),
        .onehot_out (win_grant_1h),
        .index_out  (win_grant),
        .valid_out  (win_grant_valid)
    );
    `UNUSED_VAR (win_grant_1h)

    wire [SLOT_W-1:0] ws = win_lock ? win_owner_r : win_grant;
    wire win_write = win_lock ? 1'b1 : win_grant_valid;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wctx
        // Widen BEFORE multiplying: CTX_TAG_W is only wide enough for the product, not
        // for NUM_LANES itself at one slot, so casting the operands would truncate.
        assign wctx[i] = CTX_TAG_W'((32'(ws) * NUM_LANES) + i);
    end

    wire [RTU_IDX_BITS-1:0] n_attrs = is_cand[ws] ? RTU_IDX_BITS'(RTU_RES_CAND)
                                                  : RTU_IDX_BITS'(RTU_RES_HIT);
    wire wr_payload = (wr_idx[ws] == n_attrs);
    wire wr_attr    = (wr_idx[ws] == (n_attrs + RTU_IDX_BITS'(1)));
    wire wr_status  = (wr_idx[ws] == (n_attrs + RTU_IDX_BITS'(2)));

    // Per-lane status. A candidate's non-yielding lanes are still traversing:
    // they are given PENDING so the warp keeps them in its CONTINUE loop rather
    // than exiting on a stale status (the scheduler ignores the action of a lane
    // with no candidate).
    reg [NUM_LANES-1:0][31:0] status_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (!is_cand[ws]) begin
                status_word[i] = sch_hit[wctx[i]] ? 32'(`VX_RT_STS_DONE_HIT)
                                                  : 32'(`VX_RT_STS_DONE_MISS);
            end else if (sch_ymask[wctx[i]]) begin
                status_word[i] = (sch_ycbtype[wctx[i]] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
                               ? 32'(`VX_RT_STS_YIELD_PROC)
                               : 32'(`VX_RT_STS_YIELD_ANYHIT);
            end else begin
                status_word[i] = 32'(`VX_RT_STS_PENDING);
            end
        end
    end

    // Result-span word select. The object ray is the world ray as the window
    // staged it — single-level scenes only; a TLAS walk transforms the ray per
    // instance inside the walker and would have to surface it here.
    reg [NUM_LANES-1:0][31:0] wr_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            case (wr_idx[ws])
                RTU_IDX_BITS'(0):  wr_word[i] = sch_t[wctx[i]];
                RTU_IDX_BITS'(1):  wr_word[i] = sch_u[wctx[i]];
                RTU_IDX_BITS'(2):  wr_word[i] = sch_v[wctx[i]];
                RTU_IDX_BITS'(3):  wr_word[i] = sch_prim[wctx[i]];
                RTU_IDX_BITS'(4):  wr_word[i] = sch_inst[wctx[i]];
                RTU_IDX_BITS'(5):  wr_word[i] = sch_geom[wctx[i]];
                RTU_IDX_BITS'(6):  wr_word[i] = sch_custom[wctx[i]];
                RTU_IDX_BITS'(7):  wr_word[i] = req_rays[ws][i].origin[0];
                RTU_IDX_BITS'(8):  wr_word[i] = req_rays[ws][i].origin[1];
                RTU_IDX_BITS'(9):  wr_word[i] = req_rays[ws][i].origin[2];
                RTU_IDX_BITS'(10): wr_word[i] = req_rays[ws][i].dir[0];
                RTU_IDX_BITS'(11): wr_word[i] = req_rays[ws][i].dir[1];
                RTU_IDX_BITS'(12): wr_word[i] = req_rays[ws][i].dir[2];
                RTU_IDX_BITS'(13): wr_word[i] = {{(32-RTU_CB_TYPE_BITS){1'b0}}, sch_ycbtype[wctx[i]]};
                RTU_IDX_BITS'(14): wr_word[i] = {{(32-RTU_CB_SBT_BITS){1'b0}}, sch_ysbt[wctx[i]]};
                default:           wr_word[i] = 32'd0;   // cb_handle
            endcase
            if (wr_status) begin
                wr_word[i] = status_word[i];
            end
        end
    end

    // The payload pointer is warp-uniform, and the hitAttribute belongs to every
    // lane that ran a callback, so both cover the whole trace mask.
    reg [NUM_LANES-1:0][31:0] win_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            win_word[i] = wr_word[i];
            if (wr_payload) begin
                win_word[i] = req_payload[ws];
            end else if (wr_attr) begin
                win_word[i] = res_hit_attr[ws][i];
            end
        end
    end

    assign rtu_bus_w.win_valid        = win_write;
    assign rtu_bus_w.win_data.is_cand = is_cand[ws];
    assign rtu_bus_w.win_data.wid     = req_wid[ws];
    assign rtu_bus_w.win_data.tag     = req_tag[ws];
    assign rtu_bus_w.win_data.data    = win_word;
    // A candidate's attributes only exist for its yielding lanes; its status, and
    // every whole-trace word, cover each active lane (see status_word above).
    wire wr_uniform = wr_status || wr_payload || wr_attr;
    wire [NUM_LANES-1:0] wr_cb_mask;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wcb
        assign wr_cb_mask[i] = sch_ymask[wctx[i]];
    end
    assign rtu_bus_w.win_data.mask    = (is_cand[ws] && !wr_uniform) ? wr_cb_mask : req_mask[ws];
    assign rtu_bus_w.win_data.slot    =
          wr_status  ? RTU_SLOT_BITS'(RTU_STATUS_SLOT)
        : wr_payload ? RTU_SLOT_BITS'(RTU_PAYLOAD_SLOT)
        : wr_attr    ? RTU_SLOT_BITS'(RTU_ATTR_SLOT)
                     : RTU_SLOT_BITS'(RTU_RES_BASE) + RTU_SLOT_BITS'(wr_idx[ws]);

    wire win_fire = rtu_bus_w.win_valid && rtu_bus_w.win_ready;

    // ── the CONTINUE: route it to the slot whose candidate it answers ─────
    // A warp holds one trace, so the {src, wid} that raised the candidate names
    // exactly one parked slot.
    wire [NUM_SLOTS-1:0] cont_hit_slot;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_cont_hit
        assign cont_hit_slot[s] = ((tstate[s] == T_CBWAIT) || (tstate[s] == T_CBATTR))
                               && (req_stg[s] == req_stg_idx);
    end
    wire got_cont = is_cont && (| cont_hit_slot);

    `RUNTIME_ASSERT(~(is_cont && ~got_cont),
        ("%t: *** %s: CONTINUE from staging entry %0d with no parked candidate",
            $time, INSTANCE_ID, req_stg_idx))

    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_sch_out
        assign sch_resume[s] = (tstate[s] == T_RESUME);
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane
            assign sch_action[s * NUM_LANES + i]       = cont_action[s][i];
            assign sch_action_hit_t[s * NUM_LANES + i] = cont_hit_t[s][i];
        end
    end

    integer i;
    always_ff @(posedge clk) begin
        if (reset) begin
            tstate      <= '0;   // T_IDLE
            sch_start_r <= '0;
            is_cand     <= '0;
            wr_idx      <= '0;
            stg_armed   <= '0;
            stg_full    <= '0;
            stg_bound   <= '0;
            stg_beat    <= '0;
            lstate         <= L_IDLE;
            ld_cap_vld     <= 1'b0;
            win_lock       <= 1'b0;
            slot_preloaded <= '0;
        end else begin
            sch_start_r <= '0;
            ld_cap_vld  <= 1'b0;

            // ── an arm claims its warp's staging entry ────────────────────
            // Always accepted. Only the warp-uniform half lands here; the per-lane
            // half is the RAY beats, and it goes to the RAM.
            if (arm_fire) begin
                // NOT a beat-counter reset: beats may already have overtaken this
                // arm. The counter is cleared when the ray is consumed.
                stg_armed[arm_stg_idx]   <= 1'b1;
                stg_mask[arm_stg_idx]    <= rtu_bus_w.arm_data.mask;
                stg_tag[arm_stg_idx]     <= rtu_bus_w.arm_data.tag;
                stg_payload[arm_stg_idx] <= rtu_bus_w.arm_data.payload_ptr;
                stg_scene[arm_stg_idx]   <= rtu_bus_w.arm_data.scene_base;
                stg_flags[arm_stg_idx]   <= rtu_bus_w.arm_data.flags;
                stg_cull[arm_stg_idx]    <= rtu_bus_w.arm_data.cull_mask;

                if (bind_now) begin
                    // A free slot takes it now; its beats land straight in.
                    stg_bound[arm_stg_idx] <= 1'b1;
                    stg_slot[arm_stg_idx]  <= pick_slot;

                    tstate[pick_slot]      <= T_FILL;
                    req_mask[pick_slot]    <= rtu_bus_w.arm_data.mask;
                    req_tag[pick_slot]     <= rtu_bus_w.arm_data.tag;
                    req_stg[pick_slot]     <= arm_stg_idx;
                    req_wid[pick_slot]     <= rtu_bus_w.arm_data.wid;
                    req_payload[pick_slot] <= rtu_bus_w.arm_data.payload_ptr;
                    for (i = 0; i < NUM_LANES; ++i) begin
                        // warp-uniform: broadcast the scalars to every context
                        req_rays[pick_slot][i].scene_base <= rtu_bus_w.arm_data.scene_base;
                        req_rays[pick_slot][i].flags      <= 32'(rtu_bus_w.arm_data.flags);
                        req_rays[pick_slot][i].cull_mask  <= 32'(rtu_bus_w.arm_data.cull_mask);
                        // no callback has run yet
                        cont_attr[pick_slot][i]    <= '0;
                        res_hit_attr[pick_slot][i] <= '0;
                    end
                end
            end

            // ── a RAY beat lands in its OWNER's staging entry ─────────────
            // Beats arrive in the order the TRACE burst reads its operands, so the
            // per-entry beat counter names the field and no beat carries an address.
            // The word itself is written to the RAM (see ray_ram above).
            if (is_ray) begin
                stg_beat[req_stg_idx] <= stg_beat[req_stg_idx] + RAY_IDX_W'(1);
                if (ray_bound) begin
                    // straight into the bound slot's traversal registers
                    for (i = 0; i < NUM_LANES; ++i) begin
                        case (stg_beat[req_stg_idx])
                            RAY_IDX_W'(0): req_rays[ray_slot][i].origin[0] <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(1): req_rays[ray_slot][i].origin[1] <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(2): req_rays[ray_slot][i].origin[2] <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(3): req_rays[ray_slot][i].dir[0]    <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(4): req_rays[ray_slot][i].dir[1]    <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(5): req_rays[ray_slot][i].dir[2]    <= rtu_bus_w.req_data.data[i];
                            RAY_IDX_W'(6): req_rays[ray_slot][i].t_min     <= rtu_bus_w.req_data.data[i];
                            default:       req_rays[ray_slot][i].t_max     <= rtu_bus_w.req_data.data[i];
                        endcase
                    end
                    if (stg_beat[req_stg_idx] == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                        // The ray is whole: traverse THIS cycle. No RAM, no bubble.
                        sch_start_r[ray_slot]  <= 1'b1;
                        tstate[ray_slot]       <= T_BUSY;
                        stg_bound[req_stg_idx] <= 1'b0;
                        stg_beat[req_stg_idx]  <= '0;
                    end
                end else if (stg_beat[req_stg_idx] == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    stg_full[req_stg_idx] <= 1'b1;   // every word is in; await a slot
                end
            end

            // ── the load engine ──────────────────────────────────────────
            case (lstate)
            L_IDLE: begin
                if (ld_launch) begin
                    ld_slot  <= ld_pick;
                    ld_stg_r <= pick_stg;
                    ld_issue <= RAY_IDX_W'(1);   // beat 0 is already being read
                    // Claim the entry NOW: stg_ready drops this cycle, so the next
                    // load cannot pick the same ray, and the beat counter is free to
                    // count this warp's NEXT trace (which cannot arm until this one
                    // terminates, so the reset can never race a beat).
                    stg_full[pick_stg] <= 1'b0;
                    stg_beat[pick_stg] <= '0;

                    // The target may still be writing its previous record, so ONLY the
                    // ray words may land now. Everything the outgoing record still
                    // reads — the mask, the tag, the payload, the accepted hitAttribute
                    // — is held here and committed when the slot actually starts.
                    nxt_mask[ld_pick]    <= stg_mask[pick_stg];
                    nxt_tag[ld_pick]     <= stg_tag[pick_stg];
                    nxt_stg[ld_pick]     <= pick_stg;
                    nxt_wid[ld_pick]     <= pick_stg[NW_WIDTH-1:0];  // stg = {src, wid}
                    nxt_payload[ld_pick] <= stg_payload[pick_stg];
                    for (i = 0; i < NUM_LANES; ++i) begin
                        // warp-uniform: broadcast the scalars to every context
                        req_rays[ld_pick][i].scene_base <= stg_scene[pick_stg];
                        req_rays[ld_pick][i].flags      <= 32'(stg_flags[pick_stg]);
                        req_rays[ld_pick][i].cull_mask  <= 32'(stg_cull[pick_stg]);
                    end
                    ld_cap_vld <= 1'b1;
                    ld_cap_idx <= '0;
                    lstate     <= L_STREAM;
                end
            end
            L_STREAM: begin
                // The RAM read is synchronous, so issue and capture run one cycle
                // apart: ld_issue names the word being requested, ld_cap_idx the one
                // arriving.
                ld_cap_vld <= 1'b1;
                ld_cap_idx <= ld_issue;
                if (ld_issue != RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    ld_issue <= ld_issue + RAY_IDX_W'(1);
                end
                if (ld_cap_idx == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    lstate <= L_IDLE;
                end
            end
            endcase

            if (ld_cap_vld) begin
                for (i = 0; i < NUM_LANES; ++i) begin
                    case (ld_cap_idx)
                        RAY_IDX_W'(0): req_rays[ld_slot][i].origin[0] <= ray_ram_q[i];
                        RAY_IDX_W'(1): req_rays[ld_slot][i].origin[1] <= ray_ram_q[i];
                        RAY_IDX_W'(2): req_rays[ld_slot][i].origin[2] <= ray_ram_q[i];
                        RAY_IDX_W'(3): req_rays[ld_slot][i].dir[0]    <= ray_ram_q[i];
                        RAY_IDX_W'(4): req_rays[ld_slot][i].dir[1]    <= ray_ram_q[i];
                        RAY_IDX_W'(5): req_rays[ld_slot][i].dir[2]    <= ray_ram_q[i];
                        RAY_IDX_W'(6): req_rays[ld_slot][i].t_min     <= ray_ram_q[i];
                        default:       req_rays[ld_slot][i].t_max     <= ray_ram_q[i];
                    endcase
                end
                if (ld_cap_idx == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    slot_preloaded[ld_slot] <= 1'b1;   // starts when the slot goes idle
                end
            end

            // ── a preloaded slot starts the moment it is free ────────────────
            // Its ray is already in its registers, so the whole load has been paid for
            // underneath the record write it was waiting on.
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
                if (slot_preloaded[s] && (tstate[s] == T_IDLE)) begin
                    req_mask[s]    <= nxt_mask[s];
                    req_tag[s]     <= nxt_tag[s];
                    req_stg[s]     <= nxt_stg[s];
                    req_wid[s]     <= nxt_wid[s];
                    req_payload[s] <= nxt_payload[s];
                    for (i = 0; i < NUM_LANES; ++i) begin
                        cont_attr[s][i]    <= '0;   // no callback has run yet
                        res_hit_attr[s][i] <= '0;
                    end
                    sch_start_r[s]    <= 1'b1;
                    tstate[s]         <= T_BUSY;
                    slot_preloaded[s] <= 1'b0;
                end
            end

            // ── the window-write grant ───────────────────────────────────
            if (!win_lock) begin
                if (win_grant_valid) begin
                    win_lock    <= 1'b1;
                    win_owner_r <= win_grant;
                end
            end else if (win_fire && wr_status) begin
                win_lock <= 1'b0;   // the record is whole; the channel is free
            end

            // ── the per-slot traversal FSMs ──────────────────────────────
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY: begin
                    // The record is left WHERE IT IS — in the scheduler. All this has
                    // to remember is which span to write (candidate or terminal); the
                    // words themselves are read live by the record walk above.
                    // Yield takes priority: the walk paused with a candidate.
                    if (sch_yield[s]) begin
                        is_cand[s] <= 1'b1;
                        wr_idx[s]  <= '0;
                        tstate[s]  <= T_WRITE;
                    end else if (sch_done[s]) begin
                        is_cand[s] <= 1'b0;
                        wr_idx[s]  <= '0;
                        tstate[s]  <= T_WRITE;
                    end
                end
                T_WRITE: begin
                    // Walk the record, status last: the status write is what completes
                    // the warp's parked WAIT, so the record it is about to read must
                    // already be whole. Only the slot holding the win grant advances.
                    if (win_fire && (ws == SLOT_W'(s))) begin
                        if (wr_status) begin
                            // A TERMINAL record ends the trace: release the warp's
                            // staging entry so it may arm again, and free the slot.
                            // A candidate keeps both — the same trace resumes on the
                            // CONTINUE.
                            if (!is_cand[s]) begin
                                stg_armed[req_stg[s]] <= 1'b0;
                            end
                            tstate[s] <= is_cand[s] ? T_CBWAIT : T_IDLE;
                        end else begin
                            wr_idx[s] <= wr_idx[s] + RTU_IDX_BITS'(1);
                        end
                    end
                end
                T_CBWAIT: begin
                    // The warp ran its any-hit / intersection shader and resumed the
                    // walk. An intersection shader reports its OWN t, and it rides the
                    // CONTINUE — beat 0, with the actions.
                    if (is_cont && cont_hit_slot[s]) begin
                        for (i = 0; i < NUM_LANES; ++i) begin
                            cont_action[s][i] <= rtu_bus_w.req_data.cb_action[i];
                            cont_hit_t[s][i]  <= rtu_bus_w.req_data.data[i];
                        end
                        tstate[s] <= T_CBATTR;
                    end
                end
                T_CBATTR: begin
                    // Beat 1: the shader's hitAttribute. Only the lanes that yielded a
                    // candidate have one — the rest are still traversing and their rs3
                    // is whatever the shader's register happened to hold.
                    if (is_cont && cont_hit_slot[s]) begin
                        for (i = 0; i < NUM_LANES; ++i) begin
                            if (sch_ymask[s * NUM_LANES + i]) begin
                                cont_attr[s][i] <= rtu_bus_w.req_data.data[i];
                            end
                        end
                        tstate[s] <= T_RESUME;
                    end
                end
                T_RESUME: begin
                    // sch_resume[s] is asserted for this one cycle, with the actions
                    // and the shader's t already registered. A lane that ACCEPTs binds
                    // its attribute to the hit it just committed.
                    for (i = 0; i < NUM_LANES; ++i) begin
                        if (sch_ymask[s * NUM_LANES + i]
                         && ((cont_action[s][i] == RTU_CB_ACTION_BITS'(`VX_RT_CB_ACCEPT))
                          || (cont_action[s][i] == RTU_CB_ACTION_BITS'(`VX_RT_CB_TERMINATE)))) begin
                            res_hit_attr[s][i] <= cont_attr[s][i];
                        end
                    end
                    is_cand[s] <= 1'b0;
                    tstate[s]  <= T_BUSY;
                end
                default:;   // T_IDLE / T_FILL: driven by the arm and beat paths
                endcase
            end
        end
    end


// ── RTU occupancy counters ────────────────────────────────────────────────
// Where the RTU's cycles go. The distinction that matters is IDLE_ray_waiting (idle
// with a ray in hand: our own delivery latency) vs IDLE_STARVED (idle with nothing to
// run at all: the machine in front never handed us a ray). Confusing the two is how
// the traversal engine gets optimised while starvation is the actual cost.
`ifdef DBG_RTU_OCC
    longint unsigned occ_total, occ_busy, occ_write, occ_cb, occ_fill;
    longint unsigned occ_idle_ray_waiting, occ_idle_starved, occ_traces;
    always @(posedge clk) begin
        if (reset) begin
            occ_total <= 0; occ_busy <= 0; occ_write <= 0; occ_cb <= 0;
            occ_fill <= 0; occ_idle_ray_waiting <= 0; occ_idle_starved <= 0;
            occ_traces <= 0;
        end else begin
            occ_total <= occ_total + 1;
            if (| sch_start_r) occ_traces <= occ_traces + 1;
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY:   occ_busy  <= occ_busy + 1;
                T_WRITE:  occ_write <= occ_write + 1;
                T_FILL:   occ_fill  <= occ_fill + 1;
                T_CBWAIT, T_CBATTR, T_RESUME: occ_cb <= occ_cb + 1;
                T_IDLE: begin
                    // idle WITH a ray in hand (staged or being loaded) = our own
                    // latency; idle with NOTHING to run = the SFU never delivered a
                    // ray, i.e. the RTU is STARVED by the front of the machine.
                    if ((| stg_ready) || (| slot_preloaded) || (lstate == L_STREAM))
                        occ_idle_ray_waiting <= occ_idle_ray_waiting + 1;
                    else
                        occ_idle_starved <= occ_idle_starved + 1;
                end
                default:;
                endcase
            end
        end
    end
    always @(posedge clk) begin
        if (!reset && (occ_total % 40000 == 39999)) begin
            $display("RTU-OCC @%0d: traces=%0d | busy=%0d write=%0d cb=%0d fill=%0d | IDLE_ray_waiting=%0d IDLE_STARVED=%0d",
                occ_total, occ_traces, occ_busy, occ_write, occ_cb, occ_fill,
                occ_idle_ray_waiting, occ_idle_starved);
        end
    end
`endif

endmodule
