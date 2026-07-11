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

// VX_rtu_core — cluster-shared ray-traversal engine. Accepts a warp's trace
// request (active-lane mask + per-lane ray snapshot) on the RTU bus, hands the
// whole warp to the context-pool scheduler, and returns the per-lane terminal
// status + closest-hit attributes. The scheduler walks one ray context per
// active lane concurrently over a shared datapath; node/leaf lines are fetched
// through the RTCache port, tagged by context id so responses route back.

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

    // SFU-side request / response
    VX_rtu_bus_if.slave  rtu_bus_if,

    // RTCache port
    VX_mem_bus_if.master cache_bus_if
);
    // Register the outgoing bus/cache interfaces at this module boundary so the
    // SLR-crossing seams launch/capture at flops (see VX_rtu_bus_slice). The FSM
    // drives the internal working copies; the far endpoints register the return
    // direction, so only the forward output is buffered here.
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH),
        .REQ_OUT_BUF (0),  // request already registered upstream (window/arb)
        .RSP_OUT_BUF (3)   // register our outgoing response
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

    localparam [2:0] C_IDLE    = 3'd0,
                     C_BUSY    = 3'd1,
                     C_RSP     = 3'd2,
                     C_CBYIELD = 3'd3,  // drive CB_YIELD rsp; wait rsp_ready
                     C_CBWAIT  = 3'd4;  // wait CB_ACTION req
    reg [2:0] cstate;

    // latched request + per-lane results. req_rays IS the per-context ray state
    // (it feeds sch_rays for the whole walk), so the beat-serial ingest fills it
    // in place — the bus never needs a second copy of the ray.
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [RTU_BEAT_BITS-1:0]    req_beat, rsp_beat;
    wire                       rsp_eop;
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t, res_hit_u, res_hit_v;
    reg [NUM_LANES-1:0][31:0]  res_hit_prim, res_hit_geom, res_hit_inst;
    reg [NUM_LANES-1:0][31:0]  res_hit_custom;
    reg                        sch_start;
    // latched CB_YIELD metadata (candidate attrs reuse res_hit_*).
    reg [NUM_LANES-1:0]                         cb_mask;
    reg [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]   cb_type_r;
    reg [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]    cb_sbt_r;

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

    // A ray arrives one word per beat and lands directly in the context state.
    wire req_beat_fire = rtu_bus_w.req_valid && rtu_bus_w.req_ready;

    always_ff @(posedge clk) begin
        if (reset) begin
            cstate    <= C_IDLE;
            sch_start <= 1'b0;
            req_beat  <= '0;
            rsp_beat  <= '0;
        end else begin
            sch_start <= 1'b0;
            case (cstate)
            C_IDLE: begin
                if (req_beat_fire) begin
                    if (req_beat == '0) begin
                        req_mask <= rtu_bus_w.req_data.mask;
                        req_tag  <= rtu_bus_w.req_data.tag;
                        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
                            // warp-uniform: broadcast the scalar to every context
                            req_rays[i].scene_base <= rtu_bus_w.req_data.scene_base;
                        end
                    end
                    for (integer i = 0; i < NUM_LANES; i = i + 1) begin
                        case (req_beat)
                            RTU_BEAT_BITS'(0): req_rays[i].origin[0] <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(1): req_rays[i].origin[1] <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(2): req_rays[i].origin[2] <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(3): req_rays[i].dir[0]    <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(4): req_rays[i].dir[1]    <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(5): req_rays[i].dir[2]    <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(6): req_rays[i].t_min     <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(7): req_rays[i].t_max     <= rtu_bus_w.req_data.data[i];
                            RTU_BEAT_BITS'(8): req_rays[i].flags     <= rtu_bus_w.req_data.data[i];
                            default:           req_rays[i].cull_mask <= rtu_bus_w.req_data.data[i];
                        endcase
                    end
                    if (rtu_bus_w.req_data.eop) begin
                        req_beat  <= '0;
                        sch_start <= 1'b1;
                        cstate    <= C_BUSY;
                    end else begin
                        req_beat <= req_beat + RTU_BEAT_BITS'(1);
                    end
                end
            end
            C_BUSY: begin
                // Yield takes priority: the walk paused with a candidate.
                if (sch_yield) begin
                    cb_mask <= sch_ymask[NUM_LANES-1:0];
                    for (integer i = 0; i < NUM_LANES; i = i + 1) begin
                        cb_type_r[i]    <= sch_ycbtype[i];
                        cb_sbt_r[i]     <= sch_ysbt[i];
                        // candidate attrs (res_* present the candidate at yield).
                        res_hit_t[i]    <= sch_t[i];
                        res_hit_u[i]    <= sch_u[i];
                        res_hit_v[i]    <= sch_v[i];
                        res_hit_prim[i] <= sch_prim[i];
                        res_hit_geom[i] <= sch_geom[i];
                        res_hit_inst[i] <= sch_inst[i];
                        res_hit_custom[i] <= sch_custom[i];
                    end
                    cstate <= C_CBYIELD;
                end else if (sch_done) begin
                    for (integer i = 0; i < NUM_LANES; i = i + 1) begin
                        res_status[i]   <= sch_hit[i] ? 32'(`VX_RT_STS_DONE_HIT)
                                                      : 32'(`VX_RT_STS_DONE_MISS);
                        res_hit_t[i]    <= sch_t[i];
                        res_hit_u[i]    <= sch_u[i];
                        res_hit_v[i]    <= sch_v[i];
                        res_hit_prim[i] <= sch_prim[i];
                        res_hit_geom[i] <= sch_geom[i];
                        res_hit_inst[i] <= sch_inst[i];
                        res_hit_custom[i] <= sch_custom[i];
                    end
                    cstate <= C_RSP;
                end
            end
            C_CBYIELD: begin
                // stream the CB_YIELD record; the SFU consumer acks each beat.
                if (rtu_bus_w.rsp_ready) begin
                    if (rsp_eop) begin
                        rsp_beat <= '0;
                        cstate   <= C_CBWAIT;
                    end else begin
                        rsp_beat <= rsp_beat + RTU_BEAT_BITS'(1);
                    end
                end
            end
            C_CBWAIT: begin
                // the dispatcher's CB_RET arrives as a single-beat CB_ACTION
                // request; sch_resume forwards it to the held scheduler.
                if (sch_resume) begin
                    cstate <= C_BUSY;
                end
            end
            C_RSP: begin
                if (rtu_bus_w.rsp_ready) begin
                    if (rsp_eop) begin
                        rsp_beat <= '0;
                        cstate   <= C_IDLE;
                    end else begin
                        rsp_beat <= rsp_beat + RTU_BEAT_BITS'(1);
                    end
                end
            end
            default:;
            endcase
        end
    end

    // CB_ACTION arrives in C_CBWAIT as one beat carrying the IS-computed t;
    // TRACE streams its ray in C_IDLE.
    wire req_is_cbact = (rtu_bus_w.req_data.kind == RTU_REQ_CB_ACTION);
    assign rtu_bus_w.req_ready = (cstate == C_IDLE)
                               || ((cstate == C_CBWAIT) && req_is_cbact);
    assign sch_resume = (cstate == C_CBWAIT) && rtu_bus_w.req_valid && req_is_cbact;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_sch_act
        assign sch_action[i]       = (i < NUM_LANES) ? rtu_bus_w.req_data.cb_action[i] : '0;
        assign sch_action_hit_t[i] = (i < NUM_LANES) ? rtu_bus_w.req_data.data[i]      : '0;
    end

    // ── response serializer ───────────────────────────────────────────────
    // The hit record streams out of the result registers one word per beat, so
    // no wide response payload is ever registered on the bus.
    wire is_cbyield = (cstate == C_CBYIELD);
    wire [RTU_BEAT_BITS-1:0] rsp_last = is_cbyield ? RTU_BEAT_BITS'(RTU_RSP_CB_BEATS - 1)
                                                   : RTU_BEAT_BITS'(RTU_RSP_TERM_BEATS - 1);
    assign rsp_eop = (rsp_beat == rsp_last);

    reg [NUM_LANES-1:0][31:0] rsp_word;
    always @(*) begin
        rsp_word = '0;
        for (integer i = 0; i < NUM_LANES; i = i + 1) begin
            case (rsp_beat)
                RTU_BEAT_BITS'(0): rsp_word[i] = res_hit_t[i];
                RTU_BEAT_BITS'(1): rsp_word[i] = res_hit_u[i];
                RTU_BEAT_BITS'(2): rsp_word[i] = res_hit_v[i];
                RTU_BEAT_BITS'(3): rsp_word[i] = res_hit_prim[i];
                RTU_BEAT_BITS'(4): rsp_word[i] = res_hit_inst[i];
                RTU_BEAT_BITS'(5): rsp_word[i] = res_hit_geom[i];
                RTU_BEAT_BITS'(6): rsp_word[i] = res_hit_custom[i];
                RTU_BEAT_BITS'(7): rsp_word[i] = is_cbyield
                                               ? {{(32-RTU_CB_TYPE_BITS){1'b0}}, cb_type_r[i]}
                                               : res_status[i];
                RTU_BEAT_BITS'(8): rsp_word[i] = {{(32-RTU_CB_SBT_BITS){1'b0}}, cb_sbt_r[i]};
                default:           rsp_word[i] = 32'd0;   // cb_handle
            endcase
        end
    end

    assign rtu_bus_w.rsp_valid = (cstate == C_RSP) || is_cbyield;
    assign rtu_bus_w.rsp_data.kind = is_cbyield ? RTU_RSP_CB_YIELD : RTU_RSP_TERMINAL;
    assign rtu_bus_w.rsp_data.eop  = rsp_eop;
    assign rtu_bus_w.rsp_data.tag  = req_tag;
    assign rtu_bus_w.rsp_data.data = rsp_word;
    assign rtu_bus_w.rsp_data.cb_active_mask = is_cbyield ? cb_mask : '0;

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
