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

// VX_rtu_core — socket-shared ray-traversal engine, and the MASTER of the
// graphics window (see VX_rtu_bus_if). A warp arms a TRACE and the window does
// nothing further: this core reads the ray out of the window's slot RAM one
// slot per beat, hands the whole warp to the context-pool scheduler, and writes
// the hit record back into the same slots. Node/leaf lines are fetched through
// the RTCache port, tagged by context id so responses route back.
//
// A traversal that finds a non-opaque hit writes the candidate back and parks
// at the scheduler's yield barrier. The warp reads the candidate with GETW,
// runs its any-hit / intersection shader, and resumes the walk with CONTINUE;
// the actions arrive on the `req` channel and the walk finishes into a terminal
// record. The status slot is always the LAST write of a response, because
// writing it is what completes the warp's parked WAIT.

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
    // The slot spans the walk reads and writes must each be contiguous, or the
    // base+index addressing below silently targets the wrong slots.
    `STATIC_ASSERT((`VX_RT_RAY_DIRECTION == `VX_RT_RAY_ORIGIN + 3),
        ("ray origin/direction slots must be contiguous"))
    `STATIC_ASSERT((`VX_RT_CULL_MASK == RTU_RAY_BASE + RTU_RAY_SLOTS - 1),
        ("the ray input slots must be one contiguous span"))
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
        .MST_OUT_BUF (0),  // arm/req already registered upstream (window/arb)
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
    // In-flight ray contexts, decoupled from SIMD width (VX_CFG_RTU_NUM_CTX,
    // default = NUM_LANES). The warp's NUM_LANES rays occupy the low contexts;
    // any extra contexts idle. Per-context state lives in BlockRAM (see
    // VX_rtu_scheduler), so growing NUM_CTX stays flat in fabric.
`ifndef VX_CFG_RTU_NUM_CTX
`define VX_CFG_RTU_NUM_CTX `VX_CFG_NUM_THREADS
`endif
    localparam NUM_CTX   = `VX_CFG_RTU_NUM_CTX;
    `STATIC_ASSERT((NUM_CTX >= NUM_LANES), ("VX_CFG_RTU_NUM_CTX must be >= NUM_LANES"))
    localparam CTX_TAG_W = `LOG2UP(NUM_CTX);

    localparam [2:0] C_IDLE   = 3'd0,  // await an arm
                     C_RDRAY  = 3'd1,  // read the ray out of the window
                     C_BUSY   = 3'd2,  // traversing
                     C_WRITE  = 3'd3,  // write the record back (terminal | candidate)
                     C_CBWAIT = 3'd4,  // candidate returned; await the warp's CONTINUE
                     C_RDHITT = 3'd5,  // read back the (shader-updated) HIT_T
                     C_RESUME = 3'd6;  // release the scheduler's yield barrier
    reg [2:0] cstate;

    // latched trace context. req_rays IS the per-context ray state (it feeds
    // sch_rays for the whole walk), so the slot reads fill it in place — the bus
    // never needs a second copy of the ray.
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [NW_WIDTH-1:0]         req_wid;
    reg [RTU_TB_BITS-1:0]      req_tbase;
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t, res_hit_u, res_hit_v;
    reg [NUM_LANES-1:0][31:0]  res_hit_prim, res_hit_geom, res_hit_inst;
    reg [NUM_LANES-1:0][31:0]  res_hit_custom;
    reg                        sch_start;
    // latched candidate metadata (its hit attributes reuse res_hit_*).
    reg                                         is_cand;
    reg [NUM_LANES-1:0]                         cb_mask;
    reg [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]   cb_type_r;
    reg [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]    cb_sbt_r;
    // the warp's CONTINUE actions, latched off the req channel.
    reg [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_action;
    reg [NUM_LANES-1:0][31:0]                   cont_hit_t;

    // slot read/write walk counters
    reg [RTU_IDX_BITS-1:0] rd_issue, rd_ret, wr_idx;

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
        VX_rtu_scheduler #(
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

    VX_rtu_mem #(
        .INSTANCE_ID (INSTANCE_ID),
        .TAG_WIDTH   (CTX_TAG_W)
    ) mem (
        .clk          (clk),
        .reset        (reset),
        .req_valid    (m_req_valid),
        .req_addr     (m_req_addr),
        .req_tag      (m_req_tag),
        .req_ready    (m_req_ready),
        .rsp_valid    (m_rsp_valid),
        .rsp_data     (m_rsp_data),
        .rsp_tag      (m_rsp_tag),
        .rsp_ready    (m_rsp_ready),
        .cache_bus_if (cache_bus_w)
    );

    // ── window accesses (the `win` channel) ───────────────────────────────
    // Reads walk the ray span at arm and re-fetch HIT_T after a CONTINUE (an
    // intersection shader may have written its own t there). Writes walk the
    // result span, status last.
    wire [RTU_IDX_BITS-1:0] wr_last = is_cand ? RTU_IDX_BITS'(RTU_RES_CAND)
                                              : RTU_IDX_BITS'(RTU_RES_HIT);
    wire wr_status = (wr_idx == wr_last);

    wire win_read  = ((cstate == C_RDRAY) && (rd_issue < RTU_IDX_BITS'(RTU_RAY_SLOTS)))
                  || ((cstate == C_RDHITT) && (rd_issue == '0));
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

    assign rtu_bus_w.win_valid        = win_read || win_write;
    assign rtu_bus_w.win_data.we      = win_write;
    assign rtu_bus_w.win_data.is_cand = is_cand;
    assign rtu_bus_w.win_data.wid     = req_wid;
    assign rtu_bus_w.win_data.tbase   = req_tbase;
    assign rtu_bus_w.win_data.tag     = req_tag;
    assign rtu_bus_w.win_data.data    = wr_word;
    // A candidate's attributes only exist for its yielding lanes; its status
    // covers every active lane of the trace (see status_word above).
    assign rtu_bus_w.win_data.mask    = (is_cand && !wr_status) ? cb_mask : req_mask;
    assign rtu_bus_w.win_data.slot    =
          (cstate == C_RDRAY)  ? RTU_SLOT_BITS'(RTU_RAY_BASE) + RTU_SLOT_BITS'(rd_issue)
        : (cstate == C_RDHITT) ? RTU_SLOT_BITS'(`VX_RT_HIT_T)
        : wr_status            ? RTU_SLOT_BITS'(RTU_STATUS_SLOT)
                               : RTU_SLOT_BITS'(RTU_RES_BASE) + RTU_SLOT_BITS'(wr_idx);

    wire win_fire = rtu_bus_w.win_valid && rtu_bus_w.win_ready;

    // ── the req channel: read returns and CONTINUE actions ────────────────
    // Always ready: everything arriving here is something this core is already
    // waiting for, so it can never back-pressure (which is what lets a parked
    // traversal stay reachable while another warp's arm queues).
    assign rtu_bus_w.req_ready = 1'b1;
    wire req_fire  = rtu_bus_w.req_valid;
    wire got_rdata = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_RDATA);
    wire got_cont  = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_CONT);

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
            rd_issue  <= '0;
            rd_ret    <= '0;
            wr_idx    <= '0;
        end else begin
            sch_start <= 1'b0;

            // A slot read return lands straight in the context state. Returns
            // arrive in issue order, so the return counter names the field.
            if (got_rdata) begin
                if (cstate == C_RDHITT) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        cont_hit_t[i] <= rtu_bus_w.req_data.data[i];
                    end
                end else begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        case (rd_ret)
                            RTU_IDX_BITS'(0): req_rays[i].origin[0] <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(1): req_rays[i].origin[1] <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(2): req_rays[i].origin[2] <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(3): req_rays[i].dir[0]    <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(4): req_rays[i].dir[1]    <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(5): req_rays[i].dir[2]    <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(6): req_rays[i].t_min     <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(7): req_rays[i].t_max     <= rtu_bus_w.req_data.data[i];
                            RTU_IDX_BITS'(8): req_rays[i].flags     <= rtu_bus_w.req_data.data[i];
                            default:          req_rays[i].cull_mask <= rtu_bus_w.req_data.data[i];
                        endcase
                    end
                end
                rd_ret <= rd_ret + RTU_IDX_BITS'(1);
            end

            case (cstate)
            C_IDLE: begin
                if (arm_fire) begin
                    req_mask  <= rtu_bus_w.arm_data.mask;
                    req_tag   <= rtu_bus_w.arm_data.tag;
                    req_wid   <= rtu_bus_w.arm_data.wid;
                    req_tbase <= rtu_bus_w.arm_data.tbase;
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        // warp-uniform: broadcast the scalar to every context
                        req_rays[i].scene_base <= rtu_bus_w.arm_data.scene_base;
                    end
                    rd_issue <= '0;
                    rd_ret   <= '0;
                    cstate   <= C_RDRAY;
                end
            end
            C_RDRAY: begin
                if (win_fire) begin
                    rd_issue <= rd_issue + RTU_IDX_BITS'(1);
                end
                // the ray is whole once the last slot has come back
                if (got_rdata && (rd_ret == RTU_IDX_BITS'(RTU_RAY_SLOTS - 1))) begin
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
                // Walk the result span, status last: the status write is what
                // completes the warp's parked WAIT, so the record it is about to
                // read must already be whole.
                if (win_fire) begin
                    if (wr_status) begin
                        rd_issue <= '0;
                        rd_ret   <= '0;
                        cstate   <= is_cand ? C_CBWAIT : C_IDLE;
                    end else begin
                        wr_idx <= wr_idx + RTU_IDX_BITS'(1);
                    end
                end
            end
            C_CBWAIT: begin
                // The warp ran its any-hit / intersection shader and resumed the
                // walk. An intersection shader reports its own t, so re-read
                // HIT_T rather than trusting the candidate we sent.
                if (got_cont) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        cont_action[i] <= rtu_bus_w.req_data.cb_action[i];
                    end
                    cstate <= C_RDHITT;
                end
            end
            C_RDHITT: begin
                if (win_fire) begin
                    rd_issue <= rd_issue + RTU_IDX_BITS'(1);
                end
                if (got_rdata) begin
                    cstate <= C_RESUME;
                end
            end
            C_RESUME: begin
                // sch_resume is asserted for this one cycle, with the actions and
                // the shader's t already registered.
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
