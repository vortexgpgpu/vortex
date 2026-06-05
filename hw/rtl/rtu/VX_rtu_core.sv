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
// request (active-lane mask + per-lane ray snapshot) on the RTU bus, walks
// each active lane's ray through the shared scheduler, and returns the
// per-lane terminal status + hit attributes. Node/leaf lines are fetched
// through the RTCache port. Phase 1 walks lanes serially through one
// scheduler and reports closest-hit traversal with opaque-miss leaves; the
// per-lane context pool and multiple in-flight rays arrive in Phase 3.

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
    localparam LANE_IW   = `CLOG2(NUM_LANES + 1);
    localparam LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8;

    localparam [1:0] C_IDLE   = 2'd0,
                     C_SCAN   = 2'd1,
                     C_LAUNCH = 2'd2,
                     C_BUSY   = 2'd3;
    reg [1:0] cstate;
    reg       rsp_valid_r;

    // latched request + per-lane results
    reg [NUM_LANES-1:0]        req_mask;
    rtu_ray_t [NUM_LANES-1:0]  req_rays;
    reg [TAG_WIDTH-1:0]        req_tag;
    reg [NUM_LANES-1:0][31:0]  res_status, res_hit_t;
    reg [LANE_IW-1:0]          lane_idx;
    localparam LSEL = `CLOG2(NUM_LANES);
    wire [LSEL-1:0]            lsel = lane_idx[LSEL-1:0];

    // scheduler <-> mem
    wire                              sch_start;
    wire                              sch_busy, sch_done, sch_hit;
    wire [31:0]                       sch_t, sch_nodes;
    `UNUSED_VAR (sch_busy)
    `UNUSED_VAR (sch_nodes)
    wire                              m_req_valid, m_req_ready, m_rsp_valid, m_rsp_ready;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] m_req_addr;
    wire [LINE_BITS-1:0]              m_rsp_data;

    rtu_ray_t cur_ray = req_rays[lsel];
    assign sch_start = (cstate == C_LAUNCH);

    VX_rtu_scheduler #(.INSTANCE_ID (INSTANCE_ID)) scheduler (
        .clk (clk), .reset (reset),
        .start (sch_start), .ray (cur_ray), .busy (sch_busy),
        .done (sch_done), .result_hit (sch_hit), .result_t (sch_t),
        .nodes_visited (sch_nodes),
        .mem_req_valid (m_req_valid), .mem_req_addr (m_req_addr), .mem_req_ready (m_req_ready),
        .mem_rsp_valid (m_rsp_valid), .mem_rsp_data (m_rsp_data), .mem_rsp_ready (m_rsp_ready)
    );

    VX_rtu_mem #(.INSTANCE_ID (INSTANCE_ID)) mem (
        .clk (clk), .reset (reset),
        .req_valid (m_req_valid), .req_addr (m_req_addr), .req_tag (1'b0), .req_ready (m_req_ready),
        .rsp_valid (m_rsp_valid), .rsp_data (m_rsp_data), `UNUSED_PIN (rsp_tag), .rsp_ready (m_rsp_ready),
        .cache_bus_if (cache_bus_if)
    );

    wire scan_done = (lane_idx == LANE_IW'(NUM_LANES));

    always @(posedge clk) begin
        if (reset) begin
            cstate      <= C_IDLE;
            rsp_valid_r <= 1'b0;
        end else begin
            case (cstate)
            C_IDLE: begin
                if (rtu_bus_if.req_valid) begin
                    req_mask <= rtu_bus_if.req_data.mask;
                    req_rays <= rtu_bus_if.req_data.rays;
                    req_tag  <= rtu_bus_if.req_data.tag;
                    lane_idx <= '0;
                    cstate   <= C_SCAN;
                end
            end
            C_SCAN: begin
                if (scan_done) begin
                    rsp_valid_r <= 1'b1;
                    cstate      <= C_BUSY;   // reuse C_BUSY tail as response wait
                end else if (~req_mask[lsel]) begin
                    res_status[lsel] <= '0;
                    res_hit_t[lsel]  <= '0;
                    lane_idx <= lane_idx + LANE_IW'(1);
                end else begin
                    cstate <= C_LAUNCH;
                end
            end
            C_LAUNCH: begin
                cstate <= C_BUSY;
            end
            C_BUSY: begin
                if (rsp_valid_r) begin
                    // response phase
                    if (rtu_bus_if.rsp_ready) begin
                        rsp_valid_r <= 1'b0;
                        cstate      <= C_IDLE;
                    end
                end else if (sch_done) begin
                    res_status[lsel] <= sch_hit ? 32'(`VX_RT_STS_DONE_HIT)
                                                                 : 32'(`VX_RT_STS_DONE_MISS);
                    res_hit_t[lsel]  <= sch_t;
                    lane_idx <= lane_idx + LANE_IW'(1);
                    cstate   <= C_SCAN;
                end
            end
            default:;
            endcase
        end
    end

    assign rtu_bus_if.req_ready = (cstate == C_IDLE);
    assign rtu_bus_if.rsp_valid = rsp_valid_r;
    assign rtu_bus_if.rsp_data.tag         = req_tag;
    assign rtu_bus_if.rsp_data.status      = res_status;
    assign rtu_bus_if.rsp_data.hit_t       = res_hit_t;
    assign rtu_bus_if.rsp_data.hit_u       = '0;   // Phase 1: traversal-only
    assign rtu_bus_if.rsp_data.hit_v       = '0;
    assign rtu_bus_if.rsp_data.hit_prim_id = '0;
    assign rtu_bus_if.rsp_data.hit_geometry= '0;

endmodule
