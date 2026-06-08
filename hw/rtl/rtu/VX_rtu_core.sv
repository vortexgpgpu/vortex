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
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // SFU-side request / response
    VX_rtu_bus_if.slave  rtu_bus_if,

    // RTCache port
    VX_mem_bus_if.master cache_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8;
    localparam CTX_TAG_W = `LOG2UP(NUM_LANES);

    localparam [2:0] C_IDLE    = 3'd0,
                     C_BUSY    = 3'd1,
                     C_RSP     = 3'd2,
                     C_CBYIELD = 3'd3,  // drive CB_YIELD rsp; wait rsp_ready
                     C_CBWAIT  = 3'd4;  // wait CB_ACTION req
    reg [2:0] cstate;

    // latched request + per-lane results
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t, res_hit_u, res_hit_v;
    reg [NUM_LANES-1:0][31:0]  res_hit_prim, res_hit_geom;
    reg                        sch_start;
    // latched CB_YIELD metadata (candidate attrs reuse res_hit_*).
    reg [NUM_LANES-1:0]                         cb_mask;
    reg [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]   cb_type_r;
    reg [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]    cb_sbt_r;

    // scheduler interface
    wire                              sch_busy, sch_done;
    wire [NUM_LANES-1:0]              sch_hit;
    wire [NUM_LANES-1:0][31:0]        sch_t, sch_u, sch_v, sch_prim, sch_geom;
    `UNUSED_VAR (sch_busy)
    // scheduler callback yield barrier
    wire                                       sch_yield, sch_resume;
    wire [NUM_LANES-1:0]                        sch_ymask;
    wire [NUM_LANES-1:0][RTU_CB_TYPE_BITS-1:0]  sch_ycbtype;
    wire [NUM_LANES-1:0][RTU_CB_SBT_BITS-1:0]   sch_ysbt;
    wire [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] sch_action;

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
            .NUM_CTX     (NUM_LANES)
        ) scheduler (
            .clk (clk), .reset (reset),
            .start (sch_start), .mask (req_mask), .rays (req_rays),
            .busy (sch_busy), .done (sch_done),
            .res_hit (sch_hit), .res_t (sch_t), .res_u (sch_u), .res_v (sch_v),
            .res_prim (sch_prim), .res_geom (sch_geom),
            .yield (sch_yield), .yield_mask (sch_ymask), .yield_cbtype (sch_ycbtype),
            .yield_sbt (sch_ysbt), .resume (sch_resume), .action (sch_action),
            .mem_req_valid (m_req_valid), .mem_req_addr (m_req_addr), .mem_req_tag (m_req_tag),
            .mem_req_ready (m_req_ready),
            .mem_rsp_valid (m_rsp_valid), .mem_rsp_data (m_rsp_data), .mem_rsp_tag (m_rsp_tag),
            .mem_rsp_ready (m_rsp_ready)
        );
    end else begin : g_bvh_scheduler
        VX_rtu_scheduler #(
            .INSTANCE_ID (INSTANCE_ID),
            .NUM_CTX     (NUM_LANES)
        ) scheduler (
            .clk (clk), .reset (reset),
            .start (sch_start), .mask (req_mask), .rays (req_rays),
            .busy (sch_busy), .done (sch_done),
            .res_hit (sch_hit), .res_t (sch_t), .res_u (sch_u), .res_v (sch_v),
            .res_prim (sch_prim), .res_geom (sch_geom),
            .yield (sch_yield), .yield_mask (sch_ymask), .yield_cbtype (sch_ycbtype),
            .yield_sbt (sch_ysbt), .resume (sch_resume), .action (sch_action),
            .mem_req_valid (m_req_valid), .mem_req_addr (m_req_addr), .mem_req_tag (m_req_tag),
            .mem_req_ready (m_req_ready),
            .mem_rsp_valid (m_rsp_valid), .mem_rsp_data (m_rsp_data), .mem_rsp_tag (m_rsp_tag),
            .mem_rsp_ready (m_rsp_ready)
        );
    end

    VX_rtu_mem #(.INSTANCE_ID (INSTANCE_ID), .TAG_WIDTH (CTX_TAG_W)) mem (
        .clk (clk), .reset (reset),
        .req_valid (m_req_valid), .req_addr (m_req_addr), .req_tag (m_req_tag), .req_ready (m_req_ready),
        .rsp_valid (m_rsp_valid), .rsp_data (m_rsp_data), .rsp_tag (m_rsp_tag), .rsp_ready (m_rsp_ready),
        .cache_bus_if (cache_bus_if)
    );

    always @(posedge clk) begin
        if (reset) begin
            cstate    <= C_IDLE;
            sch_start <= 1'b0;
        end else begin
            sch_start <= 1'b0;
            case (cstate)
            C_IDLE: begin
                if (rtu_bus_if.req_valid) begin
                    req_mask  <= rtu_bus_if.req_data.mask;
                    req_rays  <= rtu_bus_if.req_data.rays;
                    req_tag   <= rtu_bus_if.req_data.tag;
                    sch_start <= 1'b1;
                    cstate    <= C_BUSY;
                end
            end
            C_BUSY: begin
                // Yield takes priority: the walk paused with a candidate.
                if (sch_yield) begin
                    cb_mask <= sch_ymask;
                    for (integer i = 0; i < NUM_LANES; i = i + 1) begin
                        cb_type_r[i]    <= sch_ycbtype[i];
                        cb_sbt_r[i]     <= sch_ysbt[i];
                        // candidate attrs (res_* present the candidate at yield).
                        res_hit_t[i]    <= sch_t[i];
                        res_hit_u[i]    <= sch_u[i];
                        res_hit_v[i]    <= sch_v[i];
                        res_hit_prim[i] <= sch_prim[i];
                        res_hit_geom[i] <= sch_geom[i];
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
                    end
                    cstate <= C_RSP;
                end
            end
            C_CBYIELD: begin
                // drive the CB_YIELD rsp; the SfuUnit acks via rsp_ready.
                if (rtu_bus_if.rsp_ready) begin
                    cstate <= C_CBWAIT;
                end
            end
            C_CBWAIT: begin
                // the dispatcher's CB_RET arrives as a CB_ACTION request;
                // sch_resume forwards it to the held scheduler this cycle.
                if (sch_resume) begin
                    cstate <= C_BUSY;
                end
            end
            C_RSP: begin
                if (rtu_bus_if.rsp_ready) begin
                    cstate <= C_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    // CB_ACTION arrives in C_CBWAIT; TRACE in C_IDLE.
    wire req_is_cbact = (rtu_bus_if.req_data.kind == RTU_REQ_CBACT);
    assign rtu_bus_if.req_ready = (cstate == C_IDLE)
                               || ((cstate == C_CBWAIT) && req_is_cbact);
    assign sch_resume = (cstate == C_CBWAIT) && rtu_bus_if.req_valid && req_is_cbact;
    assign sch_action = rtu_bus_if.req_data.cb_action;

    wire is_cbyield = (cstate == C_CBYIELD);
    assign rtu_bus_if.rsp_valid = (cstate == C_RSP) || is_cbyield;
    assign rtu_bus_if.rsp_data.kind        = is_cbyield ? RTU_RSP_CBYIELD : RTU_RSP_TERMINAL;
    assign rtu_bus_if.rsp_data.tag         = req_tag;
    assign rtu_bus_if.rsp_data.status      = res_status;
    assign rtu_bus_if.rsp_data.hit_t       = res_hit_t;   // candidate t at CB_YIELD
    assign rtu_bus_if.rsp_data.hit_u       = res_hit_u;
    assign rtu_bus_if.rsp_data.hit_v       = res_hit_v;
    assign rtu_bus_if.rsp_data.hit_prim_id = res_hit_prim;
    assign rtu_bus_if.rsp_data.hit_geometry= res_hit_geom;
    assign rtu_bus_if.rsp_data.cb_active_mask = is_cbyield ? cb_mask   : '0;
    assign rtu_bus_if.rsp_data.cb_type        = is_cbyield ? cb_type_r : '0;
    assign rtu_bus_if.rsp_data.cb_sbt_idx     = is_cbyield ? cb_sbt_r  : '0;

endmodule
