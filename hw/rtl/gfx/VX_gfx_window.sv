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

// VX_gfx_window — the shared per-core graphics window SFU PE. It owns the
// per-(warp, lane) slot register file and the generic window macro-ops
// (CUSTOM1 funct3=6: SETW writes one slot, GETWF/GETW read a slot window to the
// FP/GP file). The window is reused by the FF graphics units (TEX/OM payload and
// result windows) and is present whenever any graphics extension is enabled.
//
// The RTU ray-tracing engine is a consumer of the same window: when
// VX_CFG_EXT_RTU_ENABLE is set this PE additionally services the v2 trace ISA
// (TRACE2/WAIT2/CB_RET, CUSTOM1 funct3=6/7), staging ray state into the window
// and reading hit state back out. The TRACE2 macro-op arrives pre-expanded from
// VX_gfxw_uops (one micro-op per cycle):
//   TRACE2 : CFG uop unpacks the lane-packed rs1 config + handle; ORIGIN/DIR
//            uops stream the f0..f7 ray window into the regfile; the ARM uop
//            writes tmin/tmax and launches the (blocking, single-context) walk
//            via the S_REQ/S_RSP round-trip.
//   WAIT2  : returns the latched terminal status.
//   GETWF/ : windowed regfile reads (one slot per uop) to the FP file (GETWF)
//   GETW     or GP file (GETW); SETW writes one slot (callback writeback).
// The arm op is held (execute_if.ready=0) across the trace round-trip so the bus
// request reads the assembled ray-state RF directly without latching.

`include "VX_define.vh"

module VX_gfx_window import VX_gpu_pkg::*, VX_gfx_window_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter CONS_RD_PORTS = 2
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // FF-consumer window access (the TEX/OM datapath PEs, wired in
    // VX_sfu_unit): combinational slot reads to fetch a unit's input payload,
    // plus a masked slot write to land its result. Tied off when no FF consumer
    // is present (e.g. the RTU-only config), leaving the window byte-identical.
    input  wire [NW_WIDTH-1:0]                            cons_rd_wid,
    input  wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]         cons_rd_tbase,
    input  wire [CONS_RD_PORTS-1:0][GFXW_SLOT_BITS-1:0]   cons_rd_slot,
    output wire [CONS_RD_PORTS-1:0][NUM_LANES-1:0][31:0]  cons_rd_data,
    input  wire                                           cons_wr_en,
    input  wire [NW_WIDTH-1:0]                             cons_wr_wid,
    input  wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]         cons_wr_tbase,
    input  wire [NUM_LANES-1:0]                           cons_wr_mask,
    input  wire [GFXW_SLOT_BITS-1:0]                       cons_wr_slot,
    input  wire [NUM_LANES-1:0][31:0]                      cons_wr_data,

    // FWD raster payload write port (vx_frag_fetch stages each lane's
    // frag_payload_t word into the window). A second consumer write port: it
    // targets the disjoint FRAG payload slot range, so it never collides with
    // the TEX/OM cons_wr port and needs no arbitration. Tied off when RASTER
    // is absent.
    input  wire                                           rast_wr_en,
    input  wire [NW_WIDTH-1:0]                             rast_wr_wid,
    input  wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]         rast_wr_tbase,
    input  wire [NUM_LANES-1:0]                           rast_wr_mask,
    input  wire [GFXW_SLOT_BITS-1:0]                       rast_wr_slot,
    input  wire [NUM_LANES-1:0][31:0]                      rast_wr_data

`ifdef VX_CFG_EXT_RTU_ENABLE
    ,
    // cluster-shared RTU bus
    VX_rtu_bus_if.master    rtu_bus_if,

    // shader-callback async trap raise (-> scheduler)
    VX_async_trap_if.master async_trap_if
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

`ifdef VX_CFG_EXT_RTU_ENABLE
    import VX_rtu_pkg::*;
`endif

    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);

    wire [GFXW_OP_BITS-1:0]   op    = execute_if.data.op_args.gfxw.op;
    wire [GFXW_SLOT_BITS-1:0] slot  = execute_if.data.op_args.gfxw.slot[GFXW_SLOT_BITS-1:0];
    wire [NW_WIDTH-1:0]       wid   = execute_if.data.header.wid;
    wire [PID_W-1:0]          pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0]    thread_base = THREAD_BITS'(pid) << LANE_BITS;

    // Per-(warp, lane) window register file, one 32-bit word per slot. Shared by
    // the generic ops and (under EXT_RTU) the ray-state / hit fields.
    reg [31:0] regfile [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS][GFXW_SLOT_COUNT];

    // Generic window ops. SETW writes one slot (used in both builds); the
    // GETWF/GETW reads are decoded off `op` in the result mux, and additionally
    // gate is_fastop in a pure-graphics (non-RTU) build.
    wire is_setw  = (op == GFXW_OP_SETW);

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [2:0]                uop   = execute_if.data.op_args.gfxw.uop;

    // Lane-packed config rides lanes 1..3 of the rs1 register (the implicit
    // vx_wgather layout: lane1=scene, lane2=payload, lane3=flags|cull). The v2
    // ABI requires SIMD_WIDTH >= 4; clamp the indices so narrower builds (which
    // never issue TRACE2) still elaborate.
    localparam CFG_L1 = (NUM_LANES > 1) ? 1 : 0;
    localparam CFG_L2 = (NUM_LANES > 2) ? 2 : 0;
    localparam CFG_L3 = (NUM_LANES > 3) ? 3 : 0;

    // Op classification.
    wire is_trace2 = (op == GFXW_OP_TRACE2);
    // Blocking arm: the TRACE2 ARM micro-op.
    wire is_arm = is_trace2 && (uop == GFXW_UOP_ARM);
    // Fill micro-ops that write the ray-state RF: SETW, or TRACE2 CFG/ORIGIN/DIR.
    wire is_cfg    = is_trace2 && (uop == GFXW_UOP_CFG);
    wire is_origin = is_trace2 && (uop == GFXW_UOP_ORIGIN);
    wire is_dir    = is_trace2 && (uop == GFXW_UOP_DIR);

    // Latched terminal status + scene base.
    reg [31:0] status  [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];
    reg [31:0] rt_scene[`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];

    // Async trace bus FSM (Phase 2 shader callbacks). TRACE2 blocks the warp
    // (holding it at the wait2 PC) only until the FIRST bus response:
    //   TERMINAL — opaque ray finished; latch hit window, mark terminal_ready;
    //              wait2 then completes immediately.
    //   CB_YIELD — a non-opaque candidate yielded; stage it into the regfile,
    //              raise the async trap (-> dispatcher), then service the
    //              dispatcher's CB_RET as a CB_ACTION and catch the post-resume
    //              TERMINAL in the background. wait2 (re-issued after the
    //              dispatcher's mret) blocks until that TERMINAL lands.
    localparam [2:0] B_IDLE   = 3'd0,
                     B_REQ    = 3'd1,  // drive TRACE req
                     B_RSP1   = 3'd2,  // await first rsp (TERMINAL | CB_YIELD)
                     B_ARM_WB = 3'd3,  // retire the held arm op (writeback handle)
                     B_CBRET  = 3'd4,  // await the dispatcher's CB_RET op
                     B_CBACT  = 3'd5,  // drive CB_ACTION req
                     B_RSP2   = 3'd6;  // await post-resume TERMINAL
    reg [2:0] bstate;

    // In-flight trace context (latched at arm) + callback bookkeeping.
    reg [NW_WIDTH-1:0]      if_wid;
    reg [THREAD_BITS-1:0]   if_tbase;
    reg [NUM_LANES-1:0]     if_tmask;
    reg                     yield_owed;     // first rsp was CB_YIELD
    reg [NUM_LANES-1:0]     cb_mask;        // yielding lanes
    reg [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action_lat;
    reg [NUM_LANES-1:0][31:0] cb_hit_t_lat;  // IS-computed VX_RT_HIT_T at cb_ret
    // Per-warp "terminal landed, wait2 may complete" flag.
    reg [`VX_CFG_NUM_WARPS-1:0] terminal_ready;

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
        assign ray_snap[i].scene_base= rt_scene[wid][t];
    end

    wire in_cbact = (bstate == B_CBACT);
    assign rtu_bus_if.req_valid     = (bstate == B_REQ) || in_cbact;
    assign rtu_bus_if.req_data.kind = in_cbact ? RTU_REQ_CBACT : RTU_REQ_TRACE;
    assign rtu_bus_if.req_data.mask = in_cbact ? cb_mask : if_tmask;
    assign rtu_bus_if.req_data.rays = ray_snap;                  // TRACE only
    assign rtu_bus_if.req_data.cb_action = cb_action_lat;        // CB_ACTION only
    assign rtu_bus_if.req_data.cb_hit_t  = cb_hit_t_lat;         // CB_ACTION only
    assign rtu_bus_if.req_data.tag  = ($bits(rtu_bus_if.req_data.tag))'(execute_if.data.header.uuid);
    assign rtu_bus_if.rsp_ready     = (bstate == B_RSP1) || (bstate == B_RSP2);
    `UNUSED_VAR (rtu_bus_if.rsp_data.tag)

    // ── async trap raise on CB_YIELD ───────────────────────────────────
    // Fires as the held arm op retires (B_ARM_WB), so the warp — still parked
    // at the wait2 PC — is redirected to the dispatcher with mepc = wait2 PC.
    wire armwb_fire = (bstate == B_ARM_WB) && result_if.valid && result_if.ready;
    assign async_trap_if.valid  = armwb_fire && yield_owed;   // trap entry: callback yield
    assign async_trap_if.unlock = armwb_fire;                 // resume the wstall'd trace warp
    assign async_trap_if.wid    = if_wid;
    assign async_trap_if.cause  = `VX_CFG_XLEN'(`VX_TRAP_CAUSE_RTU_CALLBACK);
    assign async_trap_if.tmask  = (`VX_CFG_NUM_THREADS'(cb_mask)) << if_tbase;

    // Op classification (callback additions).
    wire is_wait   = (op == GFXW_OP_WAIT2);
    wire is_cbret  = (op == GFXW_OP_CB_RET);
    wire is_fastop = ~is_arm && ~is_wait && ~is_cbret; // SETW/CFG/ORIGIN/DIR/GETWF/GETW

    // Op acceptance. The held arm op owns execute_if across B_REQ/B_RSP1/
    // B_ARM_WB; in every other state (B_IDLE and the in-trap B_CBRET/B_CBACT/
    // B_RSP2) execute_if is free for fast regfile ops — the callback dispatcher
    // reads its payload (GET/GETWF/GETW) before issuing cb_ret.
    wire arm_busy = (bstate == B_REQ) || (bstate == B_RSP1) || (bstate == B_ARM_WB);
    wire arm_go   = (bstate == B_IDLE) && execute_if.valid && is_arm;
    wire wait_go  = execute_if.valid && is_wait && terminal_ready[wid]; // parks until terminal
    wire cbret_go = (bstate == B_CBRET) && execute_if.valid && is_cbret;

    // Latch the per-lane active mask of the in-flight trace.
    wire [NUM_LANES-1:0] arm_lanes = execute_if.data.header.tmask[thread_base +: NUM_LANES];
`else
    `UNUSED_VAR (reset)   // no RTU FSM state to reset in a pure graphics build
    wire is_getwf = (op == GFXW_OP_GETWF);
    wire is_getw  = (op == GFXW_OP_GETW);
    wire is_fastop = is_setw || is_getwf || is_getw;
    wire arm_busy  = 1'b0;
`endif

    wire fast_go = ~arm_busy && execute_if.valid && is_fastop;

    always @(posedge clk) begin
`ifdef VX_CFG_EXT_RTU_ENABLE
        if (reset) begin
            bstate         <= B_IDLE;
            yield_owed     <= 1'b0;
            terminal_ready <= '0;
        end else begin
`endif
            // ── fast-op RF writes (SET / CFG / ORIGIN / DIR) ───────────
            if (fast_go) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (execute_if.data.header.tmask[thread_base + THREAD_BITS'(i)]) begin
                        if (is_setw) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][slot] <= execute_if.data.rs1_data[i][31:0];
                        end
`ifdef VX_CFG_EXT_RTU_ENABLE
                        if (is_cfg) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_PAYLOAD_PTR_LO] <= execute_if.data.rs1_data[CFG_L2][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_RAY_FLAGS]      <= {16'd0, execute_if.data.rs1_data[CFG_L3][15:0]};
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_CULL_MASK]      <= {16'd0, execute_if.data.rs1_data[CFG_L3][31:16]};
                            rt_scene[wid][thread_base + THREAD_BITS'(i)] <= execute_if.data.rs1_data[CFG_L1][31:0];
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
`endif
                    end
                end
            end
            // ── FF-consumer slot write (TEX/OM result into the window) ──
            // Ordered after the SETW path; in practice the two never target the
            // same (warp, slot) in the same cycle (distinct warps / program
            // order), so the priority is moot.
            if (cons_wr_en) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (cons_wr_mask[i]) begin
                        regfile[cons_wr_wid][cons_wr_tbase + THREAD_BITS'(i)][cons_wr_slot] <= cons_wr_data[i];
                    end
                end
            end
            // ── FWD raster payload write (vx_frag_fetch -> window) ──
            // Disjoint slot range from the TEX/OM cons_wr port, so program order
            // and physical slots never overlap; ordered last for a defined
            // priority if a config ever aliases them.
            if (rast_wr_en) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (rast_wr_mask[i]) begin
                        regfile[rast_wr_wid][rast_wr_tbase + THREAD_BITS'(i)][rast_wr_slot] <= rast_wr_data[i];
                    end
                end
            end
`ifdef VX_CFG_EXT_RTU_ENABLE
            // ── arm: latch in-flight context + tmin/tmax ───────────────
            if (arm_go) begin
                if_wid   <= wid;
                if_tbase <= thread_base;
                if_tmask <= arm_lanes;
                if (is_trace2) begin
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        if (execute_if.data.header.tmask[thread_base + THREAD_BITS'(i)]) begin
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_T_MIN] <= execute_if.data.rs1_data[i][31:0];
                            regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_T_MAX] <= execute_if.data.rs2_data[i][31:0];
                        end
                    end
                end
            end
            // ── cb_ret: latch the dispatcher's per-lane action ─────────
            if (cbret_go) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    cb_action_lat[i] <= execute_if.data.rs1_data[i][RTU_CB_ACTION_BITS-1:0];
                    // IS dispatcher wrote its computed t into VX_RT_HIT_T; carry
                    // it back so a PROC accept commits the IS t, not the
                    // AABB-entry candidate.
                    cb_hit_t_lat[i]  <= regfile[wid][thread_base + THREAD_BITS'(i)][`VX_RT_HIT_T];
                end
            end
            // ── wait2 retirement clears the terminal flag ──────────────
            // Gate the clear on the WAIT2 actually retiring (result_if.ready),
            // not on wait_go alone: wait_go is combinational on terminal_ready,
            // but the op only retires when the shared result port accepts it.
            // If the writeback is backpressured the cycle the terminal lands
            // (possible once multiple warps stream traces), clearing on wait_go
            // would drop the flag without retiring the WAIT2 — the warp would
            // then block forever (terminal_ready never re-set).
            if (wait_go && result_if.ready) begin
                terminal_ready[wid] <= 1'b0;
            end
            // ── bus FSM ────────────────────────────────────────────────
            case (bstate)
            B_IDLE: if (arm_go) bstate <= B_REQ;
            B_REQ:  if (rtu_bus_if.req_ready) bstate <= B_RSP1;
            B_RSP1: if (rtu_bus_if.rsp_valid) begin
                        if (rtu_bus_if.rsp_data.kind == RTU_RSP_CBYIELD) begin
                            // apply_callback_payload: stage candidate attrs.
                            for (integer i = 0; i < NUM_LANES; ++i) begin
                                if (rtu_bus_if.rsp_data.cb_active_mask[i]) begin
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_T]              <= rtu_bus_if.rsp_data.hit_t[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_U]         <= rtu_bus_if.rsp_data.hit_u[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_V]         <= rtu_bus_if.rsp_data.hit_v[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_PRIMITIVE_ID]   <= rtu_bus_if.rsp_data.hit_prim_id[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_INSTANCE_ID]    <= rtu_bus_if.rsp_data.hit_instance_id[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_GEOMETRY_INDEX] <= rtu_bus_if.rsp_data.hit_geometry[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_CB_TYPE]            <= {{(32-RTU_CB_TYPE_BITS){1'b0}}, rtu_bus_if.rsp_data.cb_type[i]};
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_SBT_IDX]        <= {{(32-RTU_CB_SBT_BITS){1'b0}}, rtu_bus_if.rsp_data.cb_sbt_idx[i]};
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_CB_HANDLE]          <= 32'd0;
                                    // Object-space ray for the IS dispatcher (vx_rt_get_objray).
                                    // Single-level (no TLAS): object ray == world ray.
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_ORIGIN + 0] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 0];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_ORIGIN + 1] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 1];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_ORIGIN + 2] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_ORIGIN + 2];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_DIRECTION + 0] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 0];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_DIRECTION + 1] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 1];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_OBJECT_RAY_DIRECTION + 2] <= regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_RAY_DIRECTION + 2];
                                end
                            end
                            cb_mask    <= rtu_bus_if.rsp_data.cb_active_mask[NUM_LANES-1:0];
                            yield_owed <= 1'b1;
                        end else begin
                            // TERMINAL: latch hit window + status.
                            for (integer i = 0; i < NUM_LANES; ++i) begin
                                if (if_tmask[i]) begin
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_T]              <= rtu_bus_if.rsp_data.hit_t[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_U]         <= rtu_bus_if.rsp_data.hit_u[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_V]         <= rtu_bus_if.rsp_data.hit_v[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_PRIMITIVE_ID]   <= rtu_bus_if.rsp_data.hit_prim_id[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_INSTANCE_ID]    <= rtu_bus_if.rsp_data.hit_instance_id[i];
                                    regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_GEOMETRY_INDEX] <= rtu_bus_if.rsp_data.hit_geometry[i];
                                    status[if_wid][if_tbase + THREAD_BITS'(i)]                             <= rtu_bus_if.rsp_data.status[i];
                                end
                            end
                            terminal_ready[if_wid] <= 1'b1;
                            yield_owed <= 1'b0;
                        end
                        bstate <= B_ARM_WB;
                    end
            B_ARM_WB: if (result_if.ready) begin
                        // handle writeback retires; async trap fired this cycle if yield.
                        yield_owed <= 1'b0;
                        bstate     <= yield_owed ? B_CBRET : B_IDLE;
                    end
            B_CBRET: if (cbret_go) bstate <= B_CBACT;
            B_CBACT: if (rtu_bus_if.req_ready) bstate <= B_RSP2;
            B_RSP2:  if (rtu_bus_if.rsp_valid) begin
                        for (integer i = 0; i < NUM_LANES; ++i) begin
                            if (if_tmask[i]) begin
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_T]              <= rtu_bus_if.rsp_data.hit_t[i];
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_U]         <= rtu_bus_if.rsp_data.hit_u[i];
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_BARY_V]         <= rtu_bus_if.rsp_data.hit_v[i];
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_PRIMITIVE_ID]   <= rtu_bus_if.rsp_data.hit_prim_id[i];
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_INSTANCE_ID]    <= rtu_bus_if.rsp_data.hit_instance_id[i];
                                regfile[if_wid][if_tbase + THREAD_BITS'(i)][`VX_RT_HIT_GEOMETRY_INDEX] <= rtu_bus_if.rsp_data.hit_geometry[i];
                                status[if_wid][if_tbase + THREAD_BITS'(i)]                             <= rtu_bus_if.rsp_data.status[i];
                            end
                        end
                        terminal_ready[if_wid] <= 1'b1;
                        bstate <= B_IDLE;
                    end
            default:;
            endcase
        end
`endif
    end

    // ── FF-consumer combinational slot reads ───────────────────────────
    // A unit fetches its per-lane input payload (e.g. vx_tex4's u,v) from a
    // contiguous slot window; the slot indices are chosen by the consumer.
    for (genvar p = 0; p < CONS_RD_PORTS; ++p) begin : g_cons_rd
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cons_rd_lane
            assign cons_rd_data[p][i] = regfile[cons_rd_wid][cons_rd_tbase + THREAD_BITS'(i)][cons_rd_slot[p]];
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
                GFXW_OP_GETWF,
                GFXW_OP_GETW:  rdata = regfile[wid][tidx][slot];
`ifdef VX_CFG_EXT_RTU_ENABLE
                GFXW_OP_WAIT2: rdata = status[wid][tidx];
                GFXW_OP_TRACE2: rdata = 32'd0;   // handle (single context)
`endif
                default:       rdata = 32'd0;   // setw / fill uops: dropped
            endcase
        end
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(rdata);
    end

    // Retirement: generic fast ops always; under EXT_RTU also wait2 when the
    // terminal landed, the dispatcher's cb_ret in B_CBRET, and the held arm op
    // in B_ARM_WB.
`ifdef VX_CFG_EXT_RTU_ENABLE
    assign result_if.valid = fast_go || wait_go || cbret_go
                          || ((bstate == B_ARM_WB) && execute_if.valid);
`else
    assign result_if.valid = fast_go;
`endif
    assign result_if.data  = rsp_data_in;
    assign execute_if.ready = result_if.valid && result_if.ready;

`ifdef DBG_TRACE_GFXW
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%t: %s gfxw-op: wid=%0d, PC=0x%0h, tmask=%b, op=%0d, slot=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, op, slot, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
