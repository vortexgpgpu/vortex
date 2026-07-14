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
// hit window (see VX_rtu_bus_if). A warp arms a TRACE and streams the ray
// straight in: the arm carries its warp-uniform half and RAY beats carry the
// per-lane half, landing directly in the per-context ray state below. Nothing
// stages it on the way, and this core never reads the window back. It hands the
// whole warp to the context-pool scheduler and writes the hit record into the
// window's result slots. Node/leaf lines are fetched through the RTCache port,
// tagged by context id so responses route back.
//
// A traversal that finds a non-opaque hit writes the candidate back and parks
// at the scheduler's yield barrier. The warp reads the candidate with GETW,
// runs its any-hit / intersection shader, and resumes the walk with CONTINUE;
// the actions arrive on the `req` channel — together with the shader's own t and
// hitAttribute, which is why the walk never has to read anything back — and it
// finishes into a terminal record. The status slot is always the LAST write of a
// response, because writing it is what completes the warp's parked WAIT.

`include "VX_define.vh"

module VX_rtu_core import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
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
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH),
        .ARM_OUT_BUF (0),  // MUST be 0: arm ready is our own idle state
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
    // In-flight ray contexts, decoupled from SIMD width. The warp's NUM_LANES
    // rays occupy the low contexts; any extra contexts idle.
    localparam NUM_CTX   = `VX_CFG_RTU_NUM_CTX;
    `STATIC_ASSERT((NUM_CTX >= NUM_LANES), ("VX_CFG_RTU_NUM_CTX must be >= NUM_LANES"))
    localparam CTX_TAG_W = `LOG2UP(NUM_CTX);

    // ── knobs this core does not implement ────────────────────────────────
    // Both are honoured by the SimX model, so a config that sets them would run
    // one machine in simulation and a different one in hardware. Fail the build
    // instead of diverging silently.
    //
    // MERGE_DEPTH — the MSHR file that merges duplicate node fetches. This core
    //   issues one request per context per fetch (per-context tags), which is
    //   exactly what MERGE_DEPTH=0 selects.
    `STATIC_ASSERT((`VX_CFG_RTU_MERGE_DEPTH == 0),
        ("VX_CFG_RTU_MERGE_DEPTH > 0 is not implemented in RTL (SimX-only); the RTL core does not merge node fetches"))

    // NUM_SLOTS — resident traces, one per core sharing this RTU. NOT implemented:
    //   this core holds ONE trace (a single ray record, a single result record, one
    //   arm-to-terminal FSM), whatever the knob says. A pool needs each of those per
    //   slot plus a slot tag on every context, so a second slot is a feature, not a
    //   parameter.
    //
    //   One slot is still CORRECT at any SOCKET_SIZE — a second core's arm simply
    //   blocks on the bus until this trace terminates, and the blocked warp is never
    //   one this RTU is waiting on. It is not correct for CYCLES: at SOCKET_SIZE > 1
    //   the SimX model overlaps the sharing cores' traces and this core serialises
    //   them, so the two machines disagree on exactly the configs that share an RTU.
    //   Closing that is the slot-pool rework.

    localparam [2:0] C_IDLE   = 3'd0,  // await an arm
                     C_RXRAY  = 3'd1,  // receive the ray the TRACE burst is streaming
                     C_BUSY   = 3'd2,  // traversing
                     C_WRITE  = 3'd3,  // write the record back (terminal | candidate)
                     C_CBWAIT = 3'd4,  // candidate returned; await the CONTINUE's t
                     C_CBATTR = 3'd5,  // ... and its hitAttribute (CONT beat 1)
                     C_RESUME = 3'd6;  // release the scheduler's yield barrier
    reg [2:0] cstate;

    // latched trace context. req_rays IS the per-context ray state (it feeds
    // sch_rays for the whole walk), so the RAY beats land in it in place — the
    // ray is never copied, staged, or stored anywhere else.
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [NW_WIDTH-1:0]         req_wid;
    reg [31:0]                 req_payload;  // warp-uniform; staged for the callbacks
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t, res_hit_u, res_hit_v;
    reg [NUM_LANES-1:0][31:0]  res_hit_prim, res_hit_geom, res_hit_inst;
    reg [NUM_LANES-1:0][31:0]  res_hit_custom;
    // The hitAttribute of the ACCEPTED candidate. Bound at the verdict, like
    // res_hit_t: a later IGNORE/DONE carries no attribute, and latching every
    // verdict's rs3 would let it overwrite the accepted one.
    reg [NUM_LANES-1:0][31:0]  res_hit_attr;
    reg                        sch_start;
    // latched candidate metadata (its hit attributes reuse res_hit_*).
    reg                                         is_cand;
    reg [NUM_LANES-1:0]                         cb_mask;
    reg [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]   cb_type_r;
    reg [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]    cb_sbt_r;
    // the warp's CONTINUE, latched off the req channel: the per-lane action, and
    // the intersection shader's own t and hitAttribute.
    reg [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_action;
    reg [NUM_LANES-1:0][31:0]                   cont_hit_t;
    reg [NUM_LANES-1:0][31:0]                   cont_attr;

    // ray-beat and record-write walk counters
    reg [RTU_IDX_BITS-1:0] ray_idx, wr_idx;

    // scheduler interface (per-context, width NUM_CTX). The warp's NUM_LANES rays
    // occupy the low contexts; the rest are masked off (start and stay idle).
    wire [NUM_CTX-1:0]               sch_mask = NUM_CTX'(req_mask);
    rtu_ray_t [NUM_CTX-1:0]          sch_rays;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_sch_rays
        assign sch_rays[i] = (i < NUM_LANES) ? req_rays[i] : '0;
    end
    wire                              sch_busy, sch_done;
    wire [NUM_CTX-1:0]               sch_hit;
    wire [NUM_CTX-1:0][31:0]         sch_t, sch_u, sch_v, sch_prim, sch_geom, sch_inst, sch_custom;
    `UNUSED_VAR (sch_busy)
    // scheduler callback yield barrier
    wire                                       sch_yield, sch_resume;
    wire [NUM_CTX-1:0]                          sch_ymask;
    wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]   sch_ycbtype;
    wire [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]    sch_ysbt;
    wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0]  sch_action;
    wire [NUM_CTX-1:0][31:0]                    sch_action_hit_t;

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
            .NUM_CTX     (NUM_CTX)
        ) scheduler (
            .clk          (clk),
            .reset        (reset),
            .start        (sch_start),
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
            .NUM_CTX     (NUM_CTX)
        ) scheduler (
            .clk          (clk),
            .reset        (reset),
            .start        (sch_start),
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

    // ── window writes (the `win` channel) ─────────────────────────────────
    // One record, written as an ordered stream: the hit attributes (plus, for a
    // candidate, the object ray and callback metadata), then the warp-uniform
    // payload pointer the callback shaders read, then the accepted hit's
    // hitAttribute, and the status LAST.
    wire [RTU_IDX_BITS-1:0] n_attrs = is_cand ? RTU_IDX_BITS'(RTU_RES_CAND)
                                              : RTU_IDX_BITS'(RTU_RES_HIT);
    wire wr_payload = (wr_idx == n_attrs);
    wire wr_attr    = (wr_idx == (n_attrs + RTU_IDX_BITS'(1)));
    wire wr_status  = (wr_idx == (n_attrs + RTU_IDX_BITS'(2)));

    wire win_write = (cstate == C_WRITE);

    // Per-lane status. A candidate's non-yielding lanes are still traversing:
    // they are given PENDING so the warp keeps them in its CONTINUE loop rather
    // than exiting on a stale status (the scheduler ignores the action of a lane
    // with no candidate).
    reg [NUM_LANES-1:0][31:0] status_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (!is_cand) begin
                status_word[i] = res_status[i];
            end else if (cb_mask[i]) begin
                status_word[i] = (cb_type_r[i] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
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
            case (wr_idx)
                RTU_IDX_BITS'(0):  wr_word[i] = res_hit_t[i];
                RTU_IDX_BITS'(1):  wr_word[i] = res_hit_u[i];
                RTU_IDX_BITS'(2):  wr_word[i] = res_hit_v[i];
                RTU_IDX_BITS'(3):  wr_word[i] = res_hit_prim[i];
                RTU_IDX_BITS'(4):  wr_word[i] = res_hit_inst[i];
                RTU_IDX_BITS'(5):  wr_word[i] = res_hit_geom[i];
                RTU_IDX_BITS'(6):  wr_word[i] = res_hit_custom[i];
                RTU_IDX_BITS'(7):  wr_word[i] = req_rays[i].origin[0];
                RTU_IDX_BITS'(8):  wr_word[i] = req_rays[i].origin[1];
                RTU_IDX_BITS'(9):  wr_word[i] = req_rays[i].origin[2];
                RTU_IDX_BITS'(10): wr_word[i] = req_rays[i].dir[0];
                RTU_IDX_BITS'(11): wr_word[i] = req_rays[i].dir[1];
                RTU_IDX_BITS'(12): wr_word[i] = req_rays[i].dir[2];
                RTU_IDX_BITS'(13): wr_word[i] = {{(32-RTU_CB_TYPE_BITS){1'b0}}, cb_type_r[i]};
                RTU_IDX_BITS'(14): wr_word[i] = {{(32-RTU_CB_SBT_BITS){1'b0}}, cb_sbt_r[i]};
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
                win_word[i] = req_payload;
            end else if (wr_attr) begin
                win_word[i] = res_hit_attr[i];
            end
        end
    end

    assign rtu_bus_w.win_valid        = win_write;
    assign rtu_bus_w.win_data.is_cand = is_cand;
    assign rtu_bus_w.win_data.wid     = req_wid;
    assign rtu_bus_w.win_data.tag     = req_tag;
    assign rtu_bus_w.win_data.data    = win_word;
    // A candidate's attributes only exist for its yielding lanes; its status, and
    // every whole-trace word, cover each active lane (see status_word above).
    wire wr_uniform = wr_status || wr_payload || wr_attr;
    assign rtu_bus_w.win_data.mask    = (is_cand && !wr_uniform) ? cb_mask : req_mask;
    assign rtu_bus_w.win_data.slot    =
          wr_status  ? RTU_SLOT_BITS'(RTU_STATUS_SLOT)
        : wr_payload ? RTU_SLOT_BITS'(RTU_PAYLOAD_SLOT)
        : wr_attr    ? RTU_SLOT_BITS'(RTU_ATTR_SLOT)
                     : RTU_SLOT_BITS'(RTU_RES_BASE) + RTU_SLOT_BITS'(wr_idx);

    wire win_fire = rtu_bus_w.win_valid && rtu_bus_w.win_ready;

    // ── the req channel: the ray, and the CONTINUE ────────────────────────
    // Always ready: everything arriving here is something this core is already
    // waiting for, so it can never back-pressure (which is what lets a parked
    // traversal stay reachable while another warp's arm queues).
    assign rtu_bus_w.req_ready = 1'b1;
    wire req_fire = rtu_bus_w.req_valid;
    wire is_ray   = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_RAY);
    wire is_cont  = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_CONT);

    // A beat is only ever taken in the state that asked for it. The arm is
    // unbuffered, so a ray cannot legally arrive before its arm is accepted, and
    // a CONTINUE cannot arrive before a candidate is out; qualify anyway and trap
    // a protocol break loudly, because the alternative is that a stray beat lands
    // in another warp's ray and the walk hangs waiting for one that never comes.
    wire got_ray  = is_ray  && (cstate == C_RXRAY);
    wire got_cont = is_cont && ((cstate == C_CBWAIT) || (cstate == C_CBATTR));

    `RUNTIME_ASSERT(~(is_ray && ~got_ray),
        ("%t: *** %s: RAY beat with no armed trace (state=%0d)", $time, INSTANCE_ID, cstate))
    `RUNTIME_ASSERT(~(is_cont && ~got_cont),
        ("%t: *** %s: CONTINUE beat with no open candidate (state=%0d)", $time, INSTANCE_ID, cstate))

    // ── the arm channel: accepted only between traces ─────────────────────
    assign rtu_bus_w.arm_ready = (cstate == C_IDLE);
    wire arm_fire = rtu_bus_w.arm_valid && rtu_bus_w.arm_ready;

    assign sch_resume = (cstate == C_RESUME);
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_sch_act
        assign sch_action[i]       = (i < NUM_LANES) ? cont_action[i] : '0;
        assign sch_action_hit_t[i] = (i < NUM_LANES) ? cont_hit_t[i]  : '0;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            cstate    <= C_IDLE;
            sch_start <= 1'b0;
            is_cand   <= 1'b0;
            ray_idx   <= '0;
            wr_idx    <= '0;
        end else begin
            sch_start <= 1'b0;

            // A RAY beat lands straight in the context's ray state. Beats arrive
            // in the order the TRACE burst reads its operands, so the beat counter
            // names the field and no beat carries an address.
            if (got_ray) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    case (ray_idx)
                        RTU_IDX_BITS'(0): req_rays[i].origin[0] <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(1): req_rays[i].origin[1] <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(2): req_rays[i].origin[2] <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(3): req_rays[i].dir[0]    <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(4): req_rays[i].dir[1]    <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(5): req_rays[i].dir[2]    <= rtu_bus_w.req_data.data[i];
                        RTU_IDX_BITS'(6): req_rays[i].t_min     <= rtu_bus_w.req_data.data[i];
                        default:          req_rays[i].t_max     <= rtu_bus_w.req_data.data[i];
                    endcase
                end
                ray_idx <= ray_idx + RTU_IDX_BITS'(1);
            end

            case (cstate)
            C_IDLE: begin
                if (arm_fire) begin
                    req_mask    <= rtu_bus_w.arm_data.mask;
                    req_tag     <= rtu_bus_w.arm_data.tag;
                    req_wid     <= rtu_bus_w.arm_data.wid;
                    req_payload <= rtu_bus_w.arm_data.payload_ptr;
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        // warp-uniform: broadcast the scalars to every context
                        req_rays[i].scene_base <= rtu_bus_w.arm_data.scene_base;
                        req_rays[i].flags      <= 32'(rtu_bus_w.arm_data.flags);
                        req_rays[i].cull_mask  <= 32'(rtu_bus_w.arm_data.cull_mask);
                        // no callback has run yet
                        cont_attr[i]    <= '0;
                        res_hit_attr[i] <= '0;
                    end
                    ray_idx <= '0;
                    cstate  <= C_RXRAY;
                end
            end
            C_RXRAY: begin
                // the ray is whole once the burst's last beat has landed
                if (got_ray && (ray_idx == RTU_IDX_BITS'(RTU_RAY_BEATS - 1))) begin
                    sch_start <= 1'b1;
                    cstate    <= C_BUSY;
                end
            end
            C_BUSY: begin
                // Yield takes priority: the walk paused with a candidate.
                if (sch_yield) begin
                    is_cand <= 1'b1;
                    cb_mask <= sch_ymask[NUM_LANES-1:0];
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        cb_type_r[i]      <= sch_ycbtype[i];
                        cb_sbt_r[i]       <= sch_ysbt[i];
                        // candidate attrs (res_* present the candidate at yield).
                        res_hit_t[i]      <= sch_t[i];
                        res_hit_u[i]      <= sch_u[i];
                        res_hit_v[i]      <= sch_v[i];
                        res_hit_prim[i]   <= sch_prim[i];
                        res_hit_geom[i]   <= sch_geom[i];
                        res_hit_inst[i]   <= sch_inst[i];
                        res_hit_custom[i] <= sch_custom[i];
                    end
                    wr_idx <= '0;
                    cstate <= C_WRITE;
                end else if (sch_done) begin
                    is_cand <= 1'b0;
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        res_status[i]     <= sch_hit[i] ? 32'(`VX_RT_STS_DONE_HIT)
                                                        : 32'(`VX_RT_STS_DONE_MISS);
                        res_hit_t[i]      <= sch_t[i];
                        res_hit_u[i]      <= sch_u[i];
                        res_hit_v[i]      <= sch_v[i];
                        res_hit_prim[i]   <= sch_prim[i];
                        res_hit_geom[i]   <= sch_geom[i];
                        res_hit_inst[i]   <= sch_inst[i];
                        res_hit_custom[i] <= sch_custom[i];
                    end
                    wr_idx <= '0;
                    cstate <= C_WRITE;
                end
            end
            C_WRITE: begin
                // Walk the record, status last: the status write is what completes
                // the warp's parked WAIT, so the record it is about to read must
                // already be whole.
                if (win_fire) begin
                    if (wr_status) begin
                        cstate <= is_cand ? C_CBWAIT : C_IDLE;
                    end else begin
                        wr_idx <= wr_idx + RTU_IDX_BITS'(1);
                    end
                end
            end
            C_CBWAIT: begin
                // The warp ran its any-hit / intersection shader and resumed the
                // walk. An intersection shader reports its OWN t, and it rides the
                // CONTINUE — beat 0, with the actions.
                if (got_cont) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        cont_action[i] <= rtu_bus_w.req_data.cb_action[i];
                        cont_hit_t[i]  <= rtu_bus_w.req_data.data[i];
                    end
                    cstate <= C_CBATTR;
                end
            end
            C_CBATTR: begin
                // Beat 1: the shader's hitAttribute. Only the lanes that yielded a
                // candidate have one — the rest are still traversing and their rs3
                // is whatever the shader's register happened to hold.
                if (got_cont) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        if (cb_mask[i]) begin
                            cont_attr[i] <= rtu_bus_w.req_data.data[i];
                        end
                    end
                    cstate <= C_RESUME;
                end
            end
            C_RESUME: begin
                // sch_resume is asserted for this one cycle, with the actions and
                // the shader's t already registered. A lane that ACCEPTs binds its
                // attribute to the hit it just committed.
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (cb_mask[i]
                     && ((cont_action[i] == RTU_CB_ACTION_BITS'(`VX_RT_CB_ACCEPT))
                      || (cont_action[i] == RTU_CB_ACTION_BITS'(`VX_RT_CB_TERMINATE)))) begin
                        res_hit_attr[i] <= cont_attr[i];
                    end
                end
                is_cand <= 1'b0;
                cstate  <= C_BUSY;
            end
            default:;
            endcase
        end
    end

    // Idle (non-lane) contexts when NUM_CTX > NUM_LANES: their per-context result
    // and yield fields are intentionally never read.
    if (NUM_CTX > NUM_LANES) begin : g_idle_ctx
        for (genvar i = NUM_LANES; i < NUM_CTX; ++i) begin : g_u
            wire _u = (|sch_hit[i]) | (|sch_t[i]) | (|sch_u[i]) | (|sch_v[i])
                    | (|sch_prim[i]) | (|sch_geom[i]) | (|sch_inst[i]) | (|sch_custom[i])
                    | (|sch_ymask[i]) | (|sch_ycbtype[i]) | (|sch_ysbt[i]);
            `UNUSED_VAR (_u)
        end
    end

endmodule
