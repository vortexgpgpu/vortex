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

// VX_rtu_unit — per-core SFU PE for the four CUSTOM1/funct3=5 ray-tracing
// ops, owning the per-(warp, lane) ray-state register file:
//   set   : latch rs1 into the addressed slot (no writeback) — 1 cycle.
//   get   : read the addressed slot back to rd — 1 cycle.
//   trace : snapshot the active lanes' ray state onto the RTU bus, wait for
//           the core's response, write the per-lane hit slots + status into
//           the RF, and return a handle. The op is held (execute_if.ready=0)
//           across the round-trip, so the bus request reads execute/RF state
//           directly without latching.
//   wait  : return the latched terminal status for the lane — 1 cycle.

`include "VX_define.vh"

module VX_rtu_unit import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // cluster-shared RTU bus
    VX_rtu_bus_if.master    rtu_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);

    `UNUSED_VAR (execute_if.data.rs2_data)
    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [RTU_SUBOP_BITS-1:0] subop = execute_if.data.op_args.rtu.subop;
    wire [RTU_SLOT_BITS-1:0]  slot  = execute_if.data.op_args.rtu.slot[RTU_SLOT_BITS-1:0];
    wire [NW_WIDTH-1:0]       wid   = execute_if.data.header.wid;
    wire [PID_W-1:0]          pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0]    thread_base = THREAD_BITS'(pid) << LANE_BITS;

    wire is_trace = (subop == RTU_SUBOP_TRACE);

    // Per-(warp, lane) ray-state RF + latched terminal status.
    reg [31:0] regfile [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS][RTU_SLOT_COUNT];
    reg [31:0] status  [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];

    localparam [1:0] S_IDLE = 2'd0, S_REQ = 2'd1, S_RSP = 2'd2, S_DONE = 2'd3;
    reg [1:0] state;

    // ── trace ray snapshot (combinational from RF + rs1 scene pointer) ─
    rtu_ray_t [NUM_LANES-1:0] ray_snap;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_snap
        wire [THREAD_BITS-1:0] t = thread_base + THREAD_BITS'(i);
        for (genvar a = 0; a < 3; ++a) begin : g_xyz
            assign ray_snap[i].origin[a] = regfile[wid][t][`VX_RT_RAY_ORIGIN + a];
            assign ray_snap[i].dir[a]    = regfile[wid][t][`VX_RT_RAY_DIRECTION + a];
        end
        assign ray_snap[i].t_min     = regfile[wid][t][`VX_RT_T_MIN];
        assign ray_snap[i].t_max     = regfile[wid][t][`VX_RT_T_MAX];
        assign ray_snap[i].flags     = regfile[wid][t][`VX_RT_RAY_FLAGS];
        assign ray_snap[i].cull_mask = regfile[wid][t][`VX_RT_CULL_MASK];
        assign ray_snap[i].scene_base= execute_if.data.rs1_data[i][31:0];
    end

    assign rtu_bus_if.req_valid     = (state == S_REQ);
    assign rtu_bus_if.req_data.mask = execute_if.data.header.tmask;
    assign rtu_bus_if.req_data.rays = ray_snap;
    assign rtu_bus_if.req_data.tag  = ($bits(rtu_bus_if.req_data.tag))'(execute_if.data.header.uuid);
    assign rtu_bus_if.rsp_ready     = (state == S_RSP);
    `UNUSED_VAR (rtu_bus_if.rsp_data.tag)

    wire set_fire = (state == S_IDLE) && execute_if.valid && execute_if.ready
                 && (subop == RTU_SUBOP_SET);

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
        end else begin
            // set: latch active lanes' slot
            if (set_fire) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (execute_if.data.header.tmask[i]) begin
                        regfile[wid][thread_base + THREAD_BITS'(i)][slot] <= execute_if.data.rs1_data[i][31:0];
                    end
                end
            end
            case (state)
            S_IDLE: begin
                if (execute_if.valid && is_trace) begin
                    state <= S_REQ;
                end
            end
            S_REQ: begin
                if (rtu_bus_if.req_ready) begin
                    state <= S_RSP;
                end
            end
            S_RSP: begin
                if (rtu_bus_if.rsp_valid) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        if (execute_if.data.header.tmask[i]) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_T]            <= rtu_bus_if.rsp_data.hit_t[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_BARY_U]       <= rtu_bus_if.rsp_data.hit_u[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_BARY_V]       <= rtu_bus_if.rsp_data.hit_v[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_PRIMITIVE_ID] <= rtu_bus_if.rsp_data.hit_prim_id[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_GEOMETRY_INDEX]<= rtu_bus_if.rsp_data.hit_geometry[i];
                            status[wid][thread_base + THREAD_BITS'(i)]                          <= rtu_bus_if.rsp_data.status[i];
                        end
                    end
                    state <= S_DONE;
                end
            end
            S_DONE: begin
                if (result_if.ready) begin
                    state <= S_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    // ── result path ───────────────────────────────────────────────────
    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        wire [THREAD_BITS-1:0] tidx = thread_base + THREAD_BITS'(i);
        reg [31:0] rdata;
        always @(*) begin
            case (subop)
                RTU_SUBOP_GET:   rdata = regfile[wid][tidx][slot];
                RTU_SUBOP_WAIT:  rdata = status[wid][tidx];
                RTU_SUBOP_TRACE: rdata = 32'd0;   // handle (single context, Phase 1)
                default:         rdata = 32'd0;   // set: dropped (no writeback)
            endcase
        end
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(rdata);
    end

    // fast ops (set/get/wait) complete in S_IDLE; trace completes in S_DONE
    wire fast_resp = (state == S_IDLE) && ~is_trace;
    assign result_if.valid = (execute_if.valid && fast_resp) || (state == S_DONE);
    assign result_if.data  = rsp_data_in;
    assign execute_if.ready = (fast_resp || (state == S_DONE)) && result_if.ready;

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%t: %s rtu-op: wid=%0d, PC=0x%0h, tmask=%b, subop=%0d, slot=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, subop, slot, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
