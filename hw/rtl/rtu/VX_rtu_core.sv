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

    localparam [1:0] C_IDLE = 2'd0,
                     C_BUSY = 2'd1,
                     C_RSP  = 2'd2;
    reg [1:0] cstate;

    // latched request + per-lane results
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t, res_hit_u, res_hit_v;
    reg [NUM_LANES-1:0][31:0]  res_hit_prim, res_hit_geom;
    reg                        sch_start;

    // scheduler interface
    wire                              sch_busy, sch_done;
    wire [NUM_LANES-1:0]              sch_hit;
    wire [NUM_LANES-1:0][31:0]        sch_t, sch_u, sch_v, sch_prim, sch_geom;
    `UNUSED_VAR (sch_busy)

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
                if (sch_done) begin
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
            C_RSP: begin
                if (rtu_bus_if.rsp_ready) begin
                    cstate <= C_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    assign rtu_bus_if.req_ready = (cstate == C_IDLE);
    assign rtu_bus_if.rsp_valid = (cstate == C_RSP);
    assign rtu_bus_if.rsp_data.tag         = req_tag;
    assign rtu_bus_if.rsp_data.status      = res_status;
    assign rtu_bus_if.rsp_data.hit_t       = res_hit_t;
    assign rtu_bus_if.rsp_data.hit_u       = res_hit_u;
    assign rtu_bus_if.rsp_data.hit_v       = res_hit_v;
    assign rtu_bus_if.rsp_data.hit_prim_id = res_hit_prim;
    assign rtu_bus_if.rsp_data.hit_geometry= res_hit_geom;

endmodule
