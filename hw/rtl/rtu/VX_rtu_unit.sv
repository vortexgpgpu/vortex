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

// VX_rtu_unit — per-core SFU PE for the ray-tracing ISA, owning the
// per-(warp, lane) ray-state / hit register file. Handles both the v1 ISA
// (CUSTOM1 funct3=5: SET/GET/TRACE/WAIT) and the v2 / v2.1 window ISA
// (funct3=6/7: TRACE2/WAIT2/GETWF/GETW). The v2 macro-ops arrive pre-expanded
// from VX_rtu_uops (one micro-op per cycle):
//   TRACE2 : CFG uop unpacks the lane-packed rs1 config + handle; ORIGIN/DIR
//            uops stream the f0..f7 ray window into the regfile; the ARM uop
//            writes tmin/tmax and launches the (blocking, single-context) walk
//            exactly like the v1 TRACE — reusing the S_REQ/S_RSP round-trip.
//   WAIT2  : returns the latched terminal status (same as v1 WAIT).
//   GETWF/ : windowed regfile reads (same as v1 GET, one slot per uop), to the
//   GETW     FP file (GETWF) or GP file (GETW).
// The op is held (execute_if.ready=0) across the trace round-trip so the bus
// request reads the assembled ray-state RF directly without latching.

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

    // Lane-packed config rides lanes 1..3 of the rs1 register (the implicit
    // vx_wgather layout: lane1=scene, lane2=payload, lane3=flags|cull). The v2
    // ABI requires SIMD_WIDTH >= 4; clamp the indices so narrower builds (which
    // never issue TRACE2) still elaborate.
    localparam CFG_L1 = (NUM_LANES > 1) ? 1 : 0;
    localparam CFG_L2 = (NUM_LANES > 2) ? 2 : 0;
    localparam CFG_L3 = (NUM_LANES > 3) ? 3 : 0;

    wire [RTU_OP_BITS-1:0]    op    = execute_if.data.op_args.rtu.op;
    wire [2:0]                uop   = execute_if.data.op_args.rtu.uop;
    wire [RTU_SLOT_BITS-1:0]  slot  = execute_if.data.op_args.rtu.slot[RTU_SLOT_BITS-1:0];
    wire                      divg  = execute_if.data.op_args.rtu.divergent;
    wire [NW_WIDTH-1:0]       wid   = execute_if.data.header.wid;
    wire [PID_W-1:0]          pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0]    thread_base = THREAD_BITS'(pid) << LANE_BITS;

    // Op classification.
    wire is_trace2 = (op == RTU_OP_TRACE2);
    // Blocking arm: v1 TRACE, or the TRACE2 ARM micro-op.
    wire is_arm = (op == RTU_OP_TRACE) || (is_trace2 && (uop == RTU_UOP_ARM));
    // Fill micro-ops that write the ray-state RF: v1 SET, or TRACE2 CFG/ORIGIN/DIR.
    wire is_cfg    = is_trace2 && (uop == RTU_UOP_CFG);
    wire is_origin = is_trace2 && (uop == RTU_UOP_ORIGIN);
    wire is_dir    = is_trace2 && (uop == RTU_UOP_DIR);

    // Per-(warp, lane) ray-state RF + latched terminal status + scene base.
    reg [31:0] regfile [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS][RTU_SLOT_COUNT];
    reg [31:0] status  [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];
    reg [31:0] rt_scene[`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];

    localparam [1:0] S_IDLE = 2'd0, S_REQ = 2'd1, S_RSP = 2'd2, S_DONE = 2'd3;
    reg [1:0] state;

    // ── trace ray snapshot (combinational from RF + scene source) ──────
    // v1 TRACE sources the scene from rs1; v2 TRACE2 sources it from the
    // CFG-staged rt_scene latch (rs1 at the ARM uop carries f6, not the config).
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
        assign ray_snap[i].scene_base= (op == RTU_OP_TRACE)
                                     ? execute_if.data.rs1_data[i][31:0]
                                     : rt_scene[wid][t];
    end

    assign rtu_bus_if.req_valid     = (state == S_REQ);
    assign rtu_bus_if.req_data.mask = execute_if.data.header.tmask;
    assign rtu_bus_if.req_data.rays = ray_snap;
    assign rtu_bus_if.req_data.tag  = ($bits(rtu_bus_if.req_data.tag))'(execute_if.data.header.uuid);
    assign rtu_bus_if.rsp_ready     = (state == S_RSP);
    `UNUSED_VAR (rtu_bus_if.rsp_data.tag)

    // A fast op is accepted in S_IDLE (everything but the blocking arm).
    wire fast_accept = (state == S_IDLE) && execute_if.valid && ~is_arm && result_if.ready;
    // The arm enters the FSM (one cycle in S_IDLE before S_REQ).
    wire arm_fire    = (state == S_IDLE) && execute_if.valid && is_arm;

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
        end else begin
            // ── fast-op RF writes ──────────────────────────────────────
            if (fast_accept) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (execute_if.data.header.tmask[i]) begin
                        if (op == RTU_OP_SET) begin
                            // v1 SET: rs1 -> addressed slot.
                            regfile[wid][thread_base + THREAD_BITS'(i)][slot] <= execute_if.data.rs1_data[i][31:0];
                        end
                        if (is_cfg) begin
                            // CFG uop: unpack lane-packed config (lanes 1..3) into
                            // the per-lane config slots; stash scene per lane.
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_PAYLOAD_PTR_LO] <= execute_if.data.rs1_data[CFG_L2][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_FLAGS]      <= {16'd0, execute_if.data.rs1_data[CFG_L3][15:0]};
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_CULL_MASK]      <= {16'd0, execute_if.data.rs1_data[CFG_L3][31:16]};
                            rt_scene[wid][thread_base + THREAD_BITS'(i)] <= divg ? execute_if.data.rs2_data[i][31:0]
                                                                                : execute_if.data.rs1_data[CFG_L1][31:0];
                        end
                        if (is_origin) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 0] <= execute_if.data.rs1_data[i][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 1] <= execute_if.data.rs2_data[i][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 2] <= execute_if.data.rs3_data[i][31:0];
                        end
                        if (is_dir) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 0] <= execute_if.data.rs1_data[i][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 1] <= execute_if.data.rs2_data[i][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 2] <= execute_if.data.rs3_data[i][31:0];
                        end
                    end
                end
            end
            // ── arm: latch tmin/tmax (TRACE2 ARM uop) on the way into S_REQ ─
            if (arm_fire && is_trace2) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (execute_if.data.header.tmask[i]) begin
                        regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_T_MIN] <= execute_if.data.rs1_data[i][31:0];
                        regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_T_MAX] <= execute_if.data.rs2_data[i][31:0];
                    end
                end
            end
            case (state)
            S_IDLE: begin
                if (execute_if.valid && is_arm) begin
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
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_T]             <= rtu_bus_if.rsp_data.hit_t[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_BARY_U]        <= rtu_bus_if.rsp_data.hit_u[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_BARY_V]        <= rtu_bus_if.rsp_data.hit_v[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_PRIMITIVE_ID]  <= rtu_bus_if.rsp_data.hit_prim_id[i];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_INSTANCE_ID]   <= 32'd0; // single-level: no TLAS instance
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_GEOMETRY_INDEX]<= rtu_bus_if.rsp_data.hit_geometry[i];
                            status[wid][thread_base + THREAD_BITS'(i)]                            <= rtu_bus_if.rsp_data.status[i];
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
            case (op)
                RTU_OP_GET,
                RTU_OP_GETWF,
                RTU_OP_GETW:  rdata = regfile[wid][tidx][slot];
                RTU_OP_WAIT,
                RTU_OP_WAIT2: rdata = status[wid][tidx];
                RTU_OP_TRACE,
                RTU_OP_TRACE2: rdata = 32'd0;   // handle (single context)
                default:       rdata = 32'd0;   // set / fill uops: dropped
            endcase
        end
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(rdata);
    end

    // fast ops complete in S_IDLE; the blocking arm completes in S_DONE.
    wire fast_resp = (state == S_IDLE) && ~is_arm;
    assign result_if.valid = (execute_if.valid && fast_resp) || (state == S_DONE);
    assign result_if.data  = rsp_data_in;
    assign execute_if.ready = (fast_resp || (state == S_DONE)) && result_if.ready;

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%t: %s rtu-op: wid=%0d, PC=0x%0h, tmask=%b, op=%0d, uop=%0d, slot=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, op, uop, slot, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
