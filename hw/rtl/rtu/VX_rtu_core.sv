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
// hit window (see VX_rtu_bus_if). Rays stage on arrival, a pool of SLOTS
// traverses several of them at once, and the scheduler's shared front end
// switches between resident contexts whenever one parks on memory.
//
// Two structures:
//
//   STAGING  one entry per {src, wid} — per warp of every core this RTU serves.
//            An arm and its RAY beats land here UNCONDITIONALLY: a warp holds
//            one trace, so its entry is free by construction. That keeps the
//            arm's ready a constant 1, so a TRACE burst can never stall in the
//            in-order SFU while holding the issue lock. The staging RAM is the
//            SINGLE home of the incoming ray: a slot holds a POINTER to its
//            entry, never a copy, and a candidate record's object-ray words are
//            read back from these same rows at write-back.
//
//   SLOTS    RTU_NUM_SLOTS traversals in flight. A slot owns NUM_LANES contexts
//            (NUM_CTX = NUM_SLOTS * NUM_LANES) inside the scheduler; here it is
//            a small descriptor: the staging pointer, the warp identity, and a
//            record-walk FSM. The fill engine claims a free slot for a staged
//            ray and streams the per-lane rays into the scheduler's ray store,
//            lane by lane — each lane starts traversing as its ray lands.
//
// The result record lives in the scheduler's window store, one field-row (all
// lanes) per word: the record write-back here is an address counter over those
// rows, plus the object-ray rows read from staging, with the status written
// LAST — writing it is what completes the warp's parked WAIT.

`include "VX_define.vh"

module VX_rtu_core import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,  // = one context per thread of a warp
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,    // warps a source core can have in flight
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

    localparam SRC_WIDTH = `UP(`CLOG2(NUM_SRCS));

    // Register the outgoing bus/cache interfaces at this module boundary so the
    // SLR-crossing seams launch/capture at flops (see VX_rtu_bus_slice).
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

    // ── the slot pool ─────────────────────────────────────────────────
    localparam NUM_SLOTS = `VX_CFG_RTU_NUM_SLOTS;
    localparam NUM_CTX   = NUM_SLOTS * NUM_LANES;
    localparam SLOT_W    = `LOG2UP(NUM_SLOTS);
    localparam CTX_TAG_W = `LOG2UP(NUM_CTX);
    localparam LANE_W    = `LOG2UP(NUM_LANES);

    `STATIC_ASSERT((`VX_CFG_RTU_NUM_CTX == NUM_CTX),
        ("VX_CFG_RTU_NUM_CTX must equal RTU_NUM_SLOTS * NUM_THREADS: a slot owns its contexts"))

    // MERGE_DEPTH — the MSHR file that merges duplicate node fetches. This core
    // issues one request per context per fetch (per-context tags), which is
    // exactly what MERGE_DEPTH=0 selects; a config that raised it would describe
    // a machine this core is not. Fail the build instead of diverging silently.
    `STATIC_ASSERT((`VX_CFG_RTU_MERGE_DEPTH == 0),
        ("VX_CFG_RTU_MERGE_DEPTH > 0 is not implemented: this core does not merge node fetches"))

    // ── ray staging: one entry per {src, wid} ─────────────────────────
    localparam NUM_STG   = NUM_SRCS * NUM_WARPS;
    localparam STG_IDX_W = `LOG2UP(NUM_STG);
    localparam RAY_IDX_W = `CLOG2(RTU_RAY_BEATS);

    reg [NUM_STG-1:0]                stg_armed;   // the arm's scalars are here
    reg [NUM_STG-1:0]                stg_full;    // all RTU_RAY_BEATS words are here
    reg [NUM_STG-1:0][RAY_IDX_W-1:0] stg_beat;    // ray beats landed so far

    // ── per-slot descriptors ──────────────────────────────────────────
    localparam [2:0] T_IDLE   = 3'd0,  // free
                     T_FILL   = 3'd1,  // the fill engine is streaming its rays
                     T_BUSY   = 3'd2,  // traversing
                     T_WRITE  = 3'd3,  // writing the record back (terminal | candidate)
                     T_CBWAIT = 3'd4,  // candidate returned; await the CONTINUE's t
                     T_CBATTR = 3'd5,  // ... and its hitAttribute (CONT beat 1)
                     T_RESUME = 3'd6,  // release this slot's yield barrier
                     T_RWAIT  = 3'd7;  // await the resume commit -> terminal record
    reg [NUM_SLOTS-1:0][2:0]           tstate;

    reg [NUM_SLOTS-1:0][NUM_LANES-1:0] req_mask;
    reg [NUM_SLOTS-1:0][TAG_WIDTH-1:0] req_tag;
    reg [NUM_SLOTS-1:0][STG_IDX_W-1:0] req_stg;      // the staging entry it launched from
    reg [NUM_SLOTS-1:0][NW_WIDTH-1:0]  req_wid;      // its warp, for the window write
    reg [NUM_SLOTS-1:0][31:0]          req_payload;  // warp-uniform; staged for the callbacks
    reg [NUM_SLOTS-1:0]                is_cand;
    reg [NUM_SLOTS-1:0][NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_action;

    // ── the scheduler ─────────────────────────────────────────────────
    wire                       ss_valid;
    wire [SLOT_W-1:0]          ss_slot;
    wire [NUM_LANES-1:0]       ss_mask;
    wire [15:0]                ss_flags, ss_cull;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] ss_scene;
    wire                       rw_valid;
    wire [CTX_TAG_W-1:0]       rw_ctx;
    wire [RTU_RAY_BEATS*32-1:0] rw_data;

    wire [NUM_SLOTS-1:0]       sch_busy, sch_done, sch_yield;
    wire [NUM_CTX-1:0]         sch_hit, sch_yld, sch_attrv;
    wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0] sch_cbtype;
    wire [NUM_SLOTS-1:0]       sch_resume;
    wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] sch_action;

    localparam WS_ADDRW = SLOT_W + RTU_WS_WORD_BITS;
    localparam ROW_BITS = NUM_LANES * 32;
    wire                 ww_valid;
    wire [WS_ADDRW-1:0]  ww_addr;
    wire [NUM_LANES-1:0] ww_wren;
    wire [ROW_BITS-1:0]  ww_data;
    wire                 wr_valid;
    wire [WS_ADDRW-1:0]  wr_addr;
    wire [ROW_BITS-1:0]  wr_data;

    wire                              m_req_valid, m_req_ready, m_rsp_valid, m_rsp_ready;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] m_req_addr;
    wire [CTX_TAG_W-1:0]              m_req_tag, m_rsp_tag;
    wire [LINE_BITS-1:0]              m_rsp_data;

    `UNUSED_VAR (sch_busy)

    VX_rtu_scheduler #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_SLOTS   (NUM_SLOTS),
        .NUM_CTX     (NUM_CTX)
    ) scheduler (
        .clk              (clk),
        .reset            (reset),
        .slot_start_valid (ss_valid),
        .slot_start_slot  (ss_slot),
        .slot_start_mask  (ss_mask),
        .slot_start_flags (ss_flags),
        .slot_start_cull  (ss_cull),
        .slot_start_scene (ss_scene),
        .ray_wr_valid     (rw_valid),
        .ray_wr_ctx       (rw_ctx),
        .ray_wr_data      (rw_data),
        .busy             (sch_busy),
        .done             (sch_done),
        .yield            (sch_yield),
        .hit_bits         (sch_hit),
        .yld_bits         (sch_yld),
        .cb_types         (sch_cbtype),
        .attr_vld         (sch_attrv),
        .resume           (sch_resume),
        .action           (sch_action),
        .win_wr_valid     (ww_valid),
        .win_wr_addr      (ww_addr),
        .win_wr_wren      (ww_wren),
        .win_wr_data      (ww_data),
        .win_rd_valid     (wr_valid),
        .win_rd_addr      (wr_addr),
        .win_rd_data      (wr_data),
        .mem_req_valid    (m_req_valid),
        .mem_req_addr     (m_req_addr),
        .mem_req_tag      (m_req_tag),
        .mem_req_ready    (m_req_ready),
        .mem_rsp_valid    (m_rsp_valid),
        .mem_rsp_data     (m_rsp_data),
        .mem_rsp_tag      (m_rsp_tag),
        .mem_rsp_ready    (m_rsp_ready)
    );

    // ── RTCache port: node/leaf fetch ─────────────────────────────────
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

    // ── the req channel: RAY beats, and the CONTINUE ──────────────────
    // Always ready: everything arriving here is something this core is already
    // waiting for, so it can never back-pressure.
    assign rtu_bus_w.req_ready = 1'b1;
    wire req_fire = rtu_bus_w.req_valid;
    wire is_ray   = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_RAY);
    wire is_cont  = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_CONT);

    wire [STG_IDX_W-1:0] req_stg_idx = STG_IDX_W'({rtu_bus_w.req_data.src, rtu_bus_w.req_data.wid});

    // ── the arm channel: ALWAYS ready ─────────────────────────────────
    assign rtu_bus_w.arm_ready = 1'b1;
    wire arm_fire = rtu_bus_w.arm_valid;
    wire [STG_IDX_W-1:0] arm_stg_idx = STG_IDX_W'({rtu_bus_w.arm_data.src, rtu_bus_w.arm_data.wid});

    // The ABI: a warp WAITs for its trace before arming another. Nothing enforces
    // it, and a violation would overwrite a live entry — corruption, not a hang.
    `RUNTIME_ASSERT(~(arm_fire && stg_armed[arm_stg_idx]),
        ("%t: *** %s: staging entry %0d armed a second trace without waiting for the first",
            $time, INSTANCE_ID, arm_stg_idx))

    `RUNTIME_ASSERT(~(is_ray && stg_full[req_stg_idx]),
        ("%t: *** %s: RAY beat for staging entry %0d whose ray is already complete",
            $time, INSTANCE_ID, req_stg_idx))

    // ── the staging RAMs ──────────────────────────────────────────────
    // The beat RAM is the single home of the per-lane ray words: the fill
    // engine reads them out to launch, and a candidate record's object-ray and
    // t_max words are read straight back at write-back. Neither reader can
    // collide with a write to the same row (the fill waits for stg_full; the
    // write-back's warp is parked in WAIT and sends no beats).
    wire                       stg_rd;
    wire [STG_IDX_W+RAY_IDX_W-1:0] stg_raddr;
    wire [NUM_LANES-1:0][31:0] stg_rdata;

    VX_dp_ram #(
        .DATAW    (NUM_LANES * 32),
        .SIZE     (NUM_STG * RTU_RAY_BEATS),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) stg_ray_ram (
        .clk   (clk),
        .reset (reset),
        .read  (stg_rd),
        .write (is_ray),
        .wren  (1'b1),
        .waddr ({req_stg_idx, stg_beat[req_stg_idx]}),
        .wdata (rtu_bus_w.req_data.data),
        .raddr (stg_raddr),
        .rdata (stg_rdata)
    );

    // warp-uniform scalars of a staged trace (read once, at the claim)
    localparam SCLW = NUM_LANES + TAG_WIDTH + 32 + `VX_CFG_MEM_ADDR_WIDTH + 16 + 16;
    wire [SCLW-1:0] stg_scl_rdata;
    wire [SLOT_W-1:0]    fill_slot_pick;
    wire [STG_IDX_W-1:0] fill_stg_pick;
    VX_dp_ram #(
        .DATAW    (SCLW),
        .SIZE     (NUM_STG),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("W")
    ) stg_scl_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (arm_fire),
        .wren  (1'b1),
        .waddr (arm_stg_idx),
        .wdata ({rtu_bus_w.arm_data.mask,
                 rtu_bus_w.arm_data.tag,
                 rtu_bus_w.arm_data.payload_ptr,
                 rtu_bus_w.arm_data.scene_base,
                 rtu_bus_w.arm_data.flags,
                 rtu_bus_w.arm_data.cull_mask}),
        .raddr (fill_stg_pick),
        .rdata (stg_scl_rdata)
    );
    wire [NUM_LANES-1:0] scl_mask;
    wire [TAG_WIDTH-1:0] scl_tag;
    wire [31:0]          scl_payload;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] scl_scene;
    wire [15:0]          scl_flags, scl_cull;
    assign {scl_mask, scl_tag, scl_payload, scl_scene, scl_flags, scl_cull} = stg_scl_rdata;

    // ── the fill engine ───────────────────────────────────────────────
    // One staged ray at a time streams into one free slot: the slot latches the
    // warp-uniform descriptor, then each active lane's 8 words are read from
    // the beat RAM, assembled, and written into the scheduler's ray store —
    // that write IS the lane's launch, so early lanes traverse under the fill.
    wire [NUM_STG-1:0] stg_ready = stg_armed & stg_full;

    wire fill_stg_valid;
    VX_priority_encoder #(
        .N (NUM_STG)
    ) stg_picker (
        .data_in    (stg_ready),
        `UNUSED_PIN (onehot_out),
        .index_out  (fill_stg_pick),
        .valid_out  (fill_stg_valid)
    );

    wire [NUM_SLOTS-1:0] slot_free;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_slot_free
        assign slot_free[s] = (tstate[s] == T_IDLE);
    end
    wire fill_slot_valid;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) slot_picker (
        .data_in    (slot_free),
        `UNUSED_PIN (onehot_out),
        .index_out  (fill_slot_pick),
        .valid_out  (fill_slot_valid)
    );

    localparam [0:0] F_IDLE = 1'd0, F_RUN = 1'd1;
    reg                  fstate;
    reg [SLOT_W-1:0]     fill_slot;
    reg [STG_IDX_W-1:0]  fill_stg;
    reg [NUM_LANES-1:0]  fill_mask;
    reg [LANE_W-1:0]     fill_lane;
    reg [RAY_IDX_W-1:0]  fill_beat;
    reg                  fill_cap_vld;
    reg [RAY_IDX_W-1:0]  fill_cap_beat;
    reg [6:0][31:0]      fill_asm;      // beats 0..6; beat 7 rides the write

    wire fill_launch = (fstate == F_IDLE) && fill_stg_valid && fill_slot_valid;

    // write-back has priority on the staging read port; the fill retries
    wire wb_stg_req;
    wire [RAY_IDX_W-1:0] wb_stg_beat;
    wire [STG_IDX_W-1:0] wb_stg_idx;

    wire fill_issue = (fstate == F_RUN) && !wb_stg_req && fill_mask[fill_lane]
                   && !(fill_cap_vld && (fill_cap_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1)));

    assign stg_rd    = wb_stg_req || fill_issue;
    assign stg_raddr = wb_stg_req ? {wb_stg_idx, wb_stg_beat} : {fill_stg, fill_beat};

    wire [31:0] fill_word = stg_rdata[fill_lane];
    wire fill_lane_done   = fill_cap_vld && (fill_cap_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1));
    wire fill_skip        = (fstate == F_RUN) && !fill_mask[fill_lane];
    wire fill_last_lane   = (fill_lane == LANE_W'(NUM_LANES - 1));

    // a lane's launch: the assembled 256-bit ray, beat 7 straight off the RAM
    assign rw_valid = fill_lane_done;
    assign rw_ctx   = CTX_TAG_W'((32'(fill_slot) * NUM_LANES) + 32'(fill_lane));
    assign rw_data  = {fill_word, fill_asm};

    // slot claim: latch the descriptor, arm the scheduler's slot
    assign ss_valid = fill_launch;
    assign ss_slot  = fill_slot_pick;
    assign ss_mask  = scl_mask;
    assign ss_flags = scl_flags;
    assign ss_cull  = scl_cull;
    assign ss_scene = scl_scene;

    // ── the CONTINUE: route it to the slot whose candidate it answers ─
    wire [NUM_SLOTS-1:0] cont_hit_slot;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_cont_hit
        assign cont_hit_slot[s] = ((tstate[s] == T_CBWAIT) || (tstate[s] == T_CBATTR))
                               && (req_stg[s] == req_stg_idx);
    end
    wire got_cont = is_cont && (| cont_hit_slot);
    `RUNTIME_ASSERT(~(is_cont && ~got_cont),
        ("%t: *** %s: CONTINUE from staging entry %0d with no parked candidate",
            $time, INSTANCE_ID, req_stg_idx))
    `UNUSED_VAR (got_cont)

    wire [SLOT_W-1:0] cont_slot;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) cont_slot_pe (
        .data_in    (cont_hit_slot),
        `UNUSED_PIN (onehot_out),
        .index_out  (cont_slot),
        `UNUSED_PIN (valid_out)
    );
    wire cont_beat0 = is_cont && (tstate[cont_slot] == T_CBWAIT);
    wire cont_beat1 = is_cont && (tstate[cont_slot] == T_CBATTR);

    // the CONTINUE's data rows land straight in the window store
    assign ww_valid = cont_beat0 || cont_beat1;
    assign ww_addr  = {cont_slot, cont_beat0 ? RTU_WS_WORD_BITS'(RTU_WS_CONT_T)
                                             : RTU_WS_WORD_BITS'(RTU_WS_CONT_ATTR)};
    assign ww_wren  = cont_beat0 ? {NUM_LANES{1'b1}}
                                 : sch_yld[32'(cont_slot)*NUM_LANES +: NUM_LANES];
    assign ww_data  = rtu_bus_w.req_data.data;

    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_resume
        assign sch_resume[s] = (tstate[s] == T_RESUME);
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane
            assign sch_action[s * NUM_LANES + i] = cont_action[s][i];
        end
    end

    // ── window-write grant (sticky for a whole record) ────────────────
    wire [NUM_SLOTS-1:0] want_win;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_want_win
        assign want_win[s] = (tstate[s] == T_WRITE);
    end

    reg              win_lock;
    reg [SLOT_W-1:0] win_owner_r;

    wire [SLOT_W-1:0] win_grant;
    wire              win_grant_valid;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) win_picker (
        .data_in    (want_win),
        `UNUSED_PIN (onehot_out),
        .index_out  (win_grant),
        .valid_out  (win_grant_valid)
    );

    wire [SLOT_W-1:0] ws = win_owner_r;

    // ── the record write-back walk ────────────────────────────────────
    // One word per step: read the field-row(s), then present the beat. The
    // sources per word index (matching the window-slot record layout):
    //   0..6   hit attrs — terminal: hit row (miss lanes: t_max / 0);
    //          candidate: yld row for candidate lanes, hit row for the rest
    //   7..12  object ray (candidate only) — the staging RAM's beat rows
    //   13,14  cb_type (flag flops), sbt (window row)
    //   15     cb_handle (0)
    //   then   payload (descriptor), hitAttribute (window row), status LAST
    localparam [2:0] WB_IDLE = 3'd0,
                     WB_RD1  = 3'd1,
                     WB_CAP1 = 3'd2,
                     WB_RD2  = 3'd3,
                     WB_CAP2 = 3'd4,
                     WB_SEND = 3'd5;
    reg [2:0]               wb_state;
    reg [RTU_IDX_BITS-1:0]  wr_idx;
    reg [ROW_BITS-1:0]      wb_d1, wb_d2;
    reg [NUM_LANES-1:0][31:0] wb_ds;

    wire [RTU_IDX_BITS-1:0] n_attrs = is_cand[ws] ? RTU_IDX_BITS'(RTU_RES_CAND)
                                                  : RTU_IDX_BITS'(RTU_RES_HIT);
    wire wr_payload = (wr_idx == n_attrs);
    wire wr_attr    = (wr_idx == (n_attrs + RTU_IDX_BITS'(1)));
    wire wr_status  = (wr_idx == (n_attrs + RTU_IDX_BITS'(2)));
    wire wr_hitspan = (wr_idx < RTU_IDX_BITS'(RTU_RES_HIT));
    wire wr_objray  = is_cand[ws] && !wr_hitspan && (wr_idx < RTU_IDX_BITS'(13));
    wire wr_cbtype  = is_cand[ws] && (wr_idx == RTU_IDX_BITS'(13));
    wire wr_sbt     = is_cand[ws] && (wr_idx == RTU_IDX_BITS'(14));

    // reads this word needs
    wire wb_need_w1 = (wb_state == WB_RD1)
                   && (wr_hitspan || wr_sbt || wr_attr) && !wr_status && !wr_payload;
    wire wb_need_w2 = is_cand[ws] && wr_hitspan;                    // the hit row
    wire wb_need_st = (wr_hitspan && (wr_idx == '0)) || wr_objray;  // t_max / object ray

    assign wr_valid = ((wb_state == WB_RD1) && wb_need_w1)
                   || ((wb_state == WB_RD2));
    assign wr_addr  = (wb_state == WB_RD2)
                    ? {ws, RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE) + RTU_WS_WORD_BITS'(wr_idx)}
                    : wr_hitspan
                        ? {ws, (is_cand[ws] ? RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE)
                                            : RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE))
                               + RTU_WS_WORD_BITS'(wr_idx)}
                    : wr_sbt  ? {ws, RTU_WS_WORD_BITS'(RTU_WS_YLD_SBT)}
                              : {ws, RTU_WS_WORD_BITS'(RTU_WS_RES_ATTR)};

    assign wb_stg_req  = (wb_state == WB_RD1) && wb_need_st;
    assign wb_stg_idx  = req_stg[ws];
    assign wb_stg_beat = wr_objray ? RAY_IDX_W'(wr_idx - RTU_IDX_BITS'(RTU_RES_HIT))
                                   : RAY_IDX_W'(RTU_RAY_BEATS - 1);   // t_max

    // per-lane word assembly (from the captured rows + the hot flags)
    wire [CTX_TAG_W-1:0] wctx [NUM_LANES];
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wctx
        assign wctx[i] = CTX_TAG_W'((32'(ws) * NUM_LANES) + i);
    end

    reg [NUM_LANES-1:0][31:0] status_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (!is_cand[ws]) begin
                status_word[i] = sch_hit[wctx[i]] ? 32'(`VX_RT_STS_DONE_HIT)
                                                  : 32'(`VX_RT_STS_DONE_MISS);
            end else if (sch_yld[wctx[i]]) begin
                status_word[i] = (sch_cbtype[wctx[i]] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
                               ? 32'(`VX_RT_STS_YIELD_PROC)
                               : 32'(`VX_RT_STS_YIELD_ANYHIT);
            end else begin
                status_word[i] = 32'(`VX_RT_STS_PENDING);
            end
        end
    end

    reg [NUM_LANES-1:0][31:0] win_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (wr_status) begin
                win_word[i] = status_word[i];
            end else if (wr_payload) begin
                win_word[i] = req_payload[ws];
            end else if (wr_attr) begin
                win_word[i] = sch_attrv[wctx[i]] ? wb_d1[i*32 +: 32] : 32'd0;
            end else if (wr_hitspan) begin
                if (is_cand[ws] && sch_yld[wctx[i]]) begin
                    win_word[i] = wb_d1[i*32 +: 32];              // the candidate row
                end else if (sch_hit[wctx[i]]) begin
                    win_word[i] = is_cand[ws] ? wb_d2[i*32 +: 32] // the hit row
                                              : wb_d1[i*32 +: 32];
                end else begin
                    // no committed hit: t reads back the ray's own t_max
                    win_word[i] = (wr_idx == '0) ? wb_ds[i] : 32'd0;
                end
            end else if (wr_objray) begin
                win_word[i] = wb_ds[i];
            end else if (wr_cbtype) begin
                win_word[i] = {{(32-RTU_CB_TYPE_BITS){1'b0}}, sch_cbtype[wctx[i]]};
            end else if (wr_sbt) begin
                win_word[i] = wb_d1[i*32 +: 32] & 32'hff;
            end else begin
                win_word[i] = 32'd0;   // cb_handle
            end
        end
    end

    wire win_send = (wb_state == WB_SEND);
    assign rtu_bus_w.win_valid        = win_send;
    assign rtu_bus_w.win_data.is_cand = is_cand[ws];
    assign rtu_bus_w.win_data.wid     = req_wid[ws];
    assign rtu_bus_w.win_data.tag     = req_tag[ws];
    assign rtu_bus_w.win_data.data    = win_word;
    // A candidate's attributes only exist for its yielding lanes; its status,
    // and every whole-trace word, cover each active lane.
    wire wr_uniform = wr_status || wr_payload || wr_attr;
    wire [NUM_LANES-1:0] wr_cb_mask = sch_yld[32'(ws)*NUM_LANES +: NUM_LANES];
    assign rtu_bus_w.win_data.mask    = (is_cand[ws] && !wr_uniform) ? wr_cb_mask : req_mask[ws];
    assign rtu_bus_w.win_data.slot    =
          wr_status  ? RTU_SLOT_BITS'(RTU_STATUS_SLOT)
        : wr_payload ? RTU_SLOT_BITS'(RTU_PAYLOAD_SLOT)
        : wr_attr    ? RTU_SLOT_BITS'(RTU_ATTR_SLOT)
                     : RTU_SLOT_BITS'(RTU_RES_BASE) + RTU_SLOT_BITS'(wr_idx);

    wire win_fire = rtu_bus_w.win_valid && rtu_bus_w.win_ready;

    // ── control ───────────────────────────────────────────────────────
    integer i;
    always_ff @(posedge clk) begin
        if (reset) begin
            tstate       <= '0;   // T_IDLE
            stg_armed    <= '0;
            stg_full     <= '0;
            stg_beat     <= '0;
            fstate       <= F_IDLE;
            fill_cap_vld <= 1'b0;
            win_lock     <= 1'b0;
            wb_state     <= WB_IDLE;
        end else begin
            fill_cap_vld <= 1'b0;

            // ── an arm claims its warp's staging entry ────────────────
            if (arm_fire) begin
                // NOT a beat-counter reset: beats may have overtaken this arm.
                stg_armed[arm_stg_idx] <= 1'b1;
            end

            // ── a RAY beat lands in its owner's staging entry ─────────
            if (is_ray) begin
                stg_beat[req_stg_idx] <= stg_beat[req_stg_idx] + RAY_IDX_W'(1);
                if (stg_beat[req_stg_idx] == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    stg_full[req_stg_idx] <= 1'b1;
                    stg_beat[req_stg_idx] <= '0;
                end
            end

            // ── the fill engine ───────────────────────────────────────
            if (fill_launch) begin
                // Claim the entry NOW: stg_ready drops this cycle so the next
                // launch cannot pick the same ray. stg_armed stays up until the
                // terminal record: it guards the entry for the whole trace.
                stg_full[fill_stg_pick]  <= 1'b0;
                fill_stg                 <= fill_stg_pick;
                fill_slot                <= fill_slot_pick;
                fill_mask                <= scl_mask;
                fill_lane                <= '0;
                fill_beat                <= '0;
                tstate[fill_slot_pick]   <= T_FILL;
                req_mask[fill_slot_pick] <= scl_mask;
                req_tag[fill_slot_pick]  <= scl_tag;
                req_stg[fill_slot_pick]  <= fill_stg_pick;
                req_wid[fill_slot_pick]  <= fill_stg_pick[NW_WIDTH-1:0];  // stg = {src, wid}
                req_payload[fill_slot_pick] <= scl_payload;
                fstate <= F_RUN;
            end else if (fstate == F_RUN) begin
                if (fill_skip || (fill_lane_done && !fill_last_lane)) begin
                    fill_lane <= fill_lane + LANE_W'(1);
                    fill_beat <= '0;
                end
                if (fill_issue) begin
                    fill_beat <= (fill_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1))
                               ? fill_beat : (fill_beat + RAY_IDX_W'(1));
                    fill_cap_vld  <= 1'b1;
                    fill_cap_beat <= fill_beat;
                end
                if (fill_cap_vld && (fill_cap_beat != RAY_IDX_W'(RTU_RAY_BEATS - 1))) begin
                    fill_asm[fill_cap_beat[2:0]] <= fill_word;
                end
                if ((fill_skip && fill_last_lane) || (fill_lane_done && fill_last_lane)) begin
                    tstate[fill_slot] <= T_BUSY;
                    fstate <= F_IDLE;
                end
            end

            // ── the CONTINUE beats ────────────────────────────────────
            if (cont_beat0) begin
                for (i = 0; i < NUM_LANES; ++i) begin
                    cont_action[cont_slot][i] <= rtu_bus_w.req_data.cb_action[i];
                end
                tstate[cont_slot] <= T_CBATTR;
            end else if (cont_beat1) begin
                tstate[cont_slot] <= T_RESUME;
            end

            // ── the per-slot record FSMs ──────────────────────────────
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY: begin
                    // yield takes priority: the walk paused with a candidate
                    if (sch_yield[s]) begin
                        is_cand[s] <= 1'b1;
                        tstate[s]  <= T_WRITE;
                    end else if (sch_done[s]) begin
                        is_cand[s] <= 1'b0;
                        tstate[s]  <= T_WRITE;
                    end
                end
                T_RESUME: begin
                    tstate[s] <= T_RWAIT;
                end
                T_RWAIT: begin
                    if (sch_done[s]) begin
                        is_cand[s] <= 1'b0;
                        tstate[s]  <= T_WRITE;
                    end
                end
                default:;
                endcase
            end

            // ── the write-back walk ───────────────────────────────────
            case (wb_state)
            WB_IDLE: begin
                if (!win_lock && win_grant_valid) begin
                    win_lock    <= 1'b1;
                    win_owner_r <= win_grant;
                    wr_idx      <= '0;
                    wb_state    <= WB_RD1;
                end
            end
            WB_RD1: begin
                // the reads (if any) were issued this cycle
                if (wb_need_w1 || wb_need_st) begin
                    wb_state <= WB_CAP1;
                end else begin
                    wb_state <= WB_SEND;
                end
            end
            WB_CAP1: begin
                wb_d1 <= wr_data;
                wb_ds <= stg_rdata;
                if (wb_need_w2) begin
                    wb_state <= WB_RD2;
                end else begin
                    wb_state <= WB_SEND;
                end
            end
            WB_RD2: begin
                wb_state <= WB_CAP2;
            end
            WB_CAP2: begin
                wb_d2    <= wr_data;
                wb_state <= WB_SEND;
            end
            WB_SEND: begin
                if (win_fire) begin
                    if (wr_status) begin
                        // the record is whole; the channel is free
                        win_lock <= 1'b0;
                        wb_state <= WB_IDLE;
                        if (is_cand[ws]) begin
                            tstate[ws] <= T_CBWAIT;
                        end else begin
                            // a TERMINAL record ends the trace: release the
                            // warp's staging entry so it may arm again
                            stg_armed[req_stg[ws]] <= 1'b0;
                            tstate[ws] <= T_IDLE;
                        end
                    end else begin
                        wr_idx   <= wr_idx + RTU_IDX_BITS'(1);
                        wb_state <= WB_RD1;
                    end
                end
            end
            default: begin
                wb_state <= WB_IDLE;
            end
            endcase
        end
    end

// ── RTU occupancy counters ────────────────────────────────────────────────
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
            if (ss_valid) begin
                occ_traces <= occ_traces + 1;
            end
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY:   occ_busy  <= occ_busy + 1;
                T_WRITE:  occ_write <= occ_write + 1;
                T_FILL:   occ_fill  <= occ_fill + 1;
                T_CBWAIT, T_CBATTR, T_RESUME, T_RWAIT: occ_cb <= occ_cb + 1;
                T_IDLE: begin
                    if ((| stg_ready) || (fstate == F_RUN)) begin
                        occ_idle_ray_waiting <= occ_idle_ray_waiting + 1;
                    end else begin
                        occ_idle_starved <= occ_idle_starved + 1;
                    end
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
