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
// Storage is a synchronous, lane-packed RAM: one word holds all NUM_LANES copies
// of a slot, addressed by {warp, simd-group, slot}. Reads take one cycle. The RAM
// is mirrored once per concurrent read port (one core-op port + CONS_RD_PORTS for
// the FF consumers); every mirror sees the same write stream. Replication buys
// read ports at block-RAM cost, which is the cheap resource here.
//
// All writers share one write port, resolved by fixed priority:
//   RAST > FSM > CONS > FILL
// The raster seed is highest so it never back-pressures the fragment
// distributor; the trace FSM outranks the execute-side fill so a parked warp's
// hit burst always drains; SETW/fill is last because it can stall its own warp.
// Multi-slot writers (the TRACE2 fill uops, the hit-record burst, the object-ray
// copy) present one word per cycle and advance only on grant.
//
// The RTU ray-tracing engine is a consumer of the same window: when
// VX_CFG_EXT_RTU_ENABLE is set this PE additionally services the v2 trace ISA
// (TRACE2/WAIT2/CB_RET, CUSTOM1 funct3=6/7), staging ray state into the window
// and reading hit state back out. The TRACE2 macro-op arrives pre-expanded from
// VX_gfxw_uops (one micro-op per cycle):
//   TRACE2 : CFG uop unpacks the lane-packed rs1 config + handle; ORIGIN/DIR
//            uops stream the f0..f7 ray window into the RAM; the ARM uop writes
//            tmin/tmax, then walks the assembled ray back out of the RAM one
//            word per cycle into the bus request register and launches the
//            (blocking, single-context) traversal.
//   WAIT2  : returns the latched terminal status.
//   GETWF/ : windowed reads (one slot per uop) to the FP file (GETWF)
//   GETW     or GP file (GETW); SETW writes one slot (callback writeback).
// The arm op is held (execute_if.ready=0) across the trace round-trip.

`include "VX_define.vh"

module VX_gfx_window import VX_gpu_pkg::*, VX_gfx_window_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter CONS_RD_PORTS = 2,
    parameter RTU_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // FF-consumer window access (the TEX/OM datapath PEs, wired in
    // VX_sfu_unit): synchronous slot reads to fetch a unit's input payload,
    // plus a masked slot write to land its result. Tied off when no FF consumer
    // is present (e.g. the RTU-only config), leaving the window byte-identical.
    VX_gfx_win_rd_if.slave                                cons_rd_if,
    VX_gfx_win_wr_if.slave                                   cons_wr_if,

    // FWD raster payload write port (the raster distributor stages each lane's
    // frag_payload_t word into the window). Highest write priority, so its
    // `ready` is constant and the distributor never stalls.
    VX_gfx_win_wr_if.slave                                    rast_wr_if

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

    localparam WIN_GROUPS  = `VX_CFG_NUM_THREADS / NUM_LANES;
    localparam RAM_SIZE    = `VX_CFG_NUM_WARPS * WIN_GROUPS * GFXW_SLOT_COUNT;
    localparam RAM_ADDRW   = `CLOG2(RAM_SIZE);
    localparam RAM_DATAW   = 32 * NUM_LANES;
    localparam RD_PORTS    = CONS_RD_PORTS + 1;   // + the core-op port

    // The address packs {warp, simd-group, slot}; both strides must be powers of
    // two or the concatenation below degenerates into a multiplier.
    `STATIC_ASSERT(((1 << GFXW_SLOT_BITS) == GFXW_SLOT_COUNT),
        ("window slot count must be a power of two"))
    `STATIC_ASSERT(((1 << LANE_BITS) == NUM_LANES),
        ("window lane count must be a power of two"))

    // Address a slot of one (warp, simd-group). `tbase` is a thread index, so its
    // low LANE_BITS are zero and the simd-group is the remaining high bits.
    function automatic [RAM_ADDRW-1:0] win_addr (
        input [NW_WIDTH-1:0]       w,
        input [THREAD_BITS-1:0]    tb,
        input [GFXW_SLOT_BITS-1:0] s
    );
        reg [RAM_ADDRW-1:0] grp;
        begin
            grp = (RAM_ADDRW'(w) * RAM_ADDRW'(WIN_GROUPS)) + RAM_ADDRW'(tb >> LANE_BITS);
            win_addr = (grp * RAM_ADDRW'(GFXW_SLOT_COUNT)) + RAM_ADDRW'(s);
        end
    endfunction

    wire [GFXW_OP_BITS-1:0]   op    = execute_if.data.op_args.gfxw.op;
    wire [GFXW_SLOT_BITS-1:0] slot  = execute_if.data.op_args.gfxw.slot[GFXW_SLOT_BITS-1:0];
    wire [NW_WIDTH-1:0]       wid   = execute_if.data.header.wid;
    wire [PID_W-1:0]          pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0]    thread_base = THREAD_BITS'(pid) << LANE_BITS;

    // ── the single write port ──────────────────────────────────────────────
    typedef struct packed {
        logic [NW_WIDTH-1:0]        wid;
        logic [THREAD_BITS-1:0]     tbase;
        logic [GFXW_SLOT_BITS-1:0]  slot;
        logic [NUM_LANES-1:0]       mask;
        logic [NUM_LANES-1:0][31:0] data;
    } win_wr_t;

    win_wr_t wr_rast, wr_fsm, wr_cons, wr_fill;
    wire     req_rast, req_fsm, req_cons, req_fill;
    wire     gnt_rast, gnt_fsm, gnt_cons, gnt_fill;

    assign gnt_rast = req_rast;
    assign gnt_fsm  = req_fsm  && ~req_rast;
    assign gnt_cons = req_cons && ~req_rast && ~req_fsm;
    assign gnt_fill = req_fill && ~req_rast && ~req_fsm && ~req_cons;

    win_wr_t wr_sel;
    always @(*) begin
        case (1'b1)
            gnt_rast: wr_sel = wr_rast;
            gnt_fsm:  wr_sel = wr_fsm;
            gnt_cons: wr_sel = wr_cons;
            default:  wr_sel = wr_fill;
        endcase
    end

    wire                 ram_write = req_rast || req_fsm || req_cons || req_fill;
    wire [RAM_ADDRW-1:0] ram_waddr = win_addr(wr_sel.wid, wr_sel.tbase, wr_sel.slot);
    wire [NUM_LANES-1:0] ram_wren  = wr_sel.mask;
    wire [RAM_DATAW-1:0] ram_wdata = wr_sel.data;

    // ── read ports: one core-op port + the FF-consumer ports ───────────────
    wire [RD_PORTS-1:0][RAM_ADDRW-1:0] ram_raddr;
    wire [RD_PORTS-1:0]                ram_rden;
    wire [RD_PORTS-1:0][RAM_DATAW-1:0] ram_rdata;

    // One mirror per read port. VX_dp_ram is 1W1R; replication is how a register
    // file buys read ports once its storage is a memory.
    for (genvar p = 0; p < RD_PORTS; ++p) begin : g_win_ram
        VX_dp_ram #(
            .DATAW    (RAM_DATAW),
            .SIZE     (RAM_SIZE),
            .WRENW    (NUM_LANES),
            .OUT_REG  (1),
            .RDW_MODE ("R")
        ) win_ram (
            .clk   (clk),
            .reset (reset),
            .read  (ram_rden[p]),
            .write (ram_write),
            .wren  (ram_wren),
            .waddr (ram_waddr),
            .wdata (ram_wdata),
            .raddr (ram_raddr[p]),
            .rdata (ram_rdata[p])
        );
    end

    // Core-op port (mirror 0): the result mux, the ARM ray walk, the object-ray
    // copy and CB_RET's hit-t fetch. Mutually exclusive by FSM state.
    wire [RAM_ADDRW-1:0]       core_raddr;
    wire                       core_rden;
    wire [NUM_LANES-1:0][31:0] core_rdata = ram_rdata[0];
    assign ram_raddr[0] = core_raddr;
    assign ram_rden[0]  = core_rden;

    // FF-consumer ports (mirrors 1..CONS_RD_PORTS). Free-running: the consumer
    // drives its slot and samples the word one cycle later.
    for (genvar p = 0; p < CONS_RD_PORTS; ++p) begin : g_cons_rd
        assign ram_raddr[p + 1] = win_addr(cons_rd_if.req.wid, cons_rd_if.req.tbase, cons_rd_if.req.slot[p]);
        assign ram_rden[p + 1]  = 1'b1;
        assign cons_rd_if.data[p] = ram_rdata[p + 1];
    end

    // ── raster seed (top priority: constant ready) ─────────────────────────
    assign req_rast    = rast_wr_if.valid;
    assign wr_rast.wid   = rast_wr_if.data.wid;
    assign wr_rast.tbase = rast_wr_if.data.tbase;
    assign wr_rast.slot  = rast_wr_if.data.slot;
    assign wr_rast.mask  = rast_wr_if.data.mask;
    assign wr_rast.data  = rast_wr_if.data.data;
    assign rast_wr_if.ready = 1'b1;

    // ── FF-consumer result write (TEX texel) ───────────────────────────────
    assign req_cons    = cons_wr_if.valid;
    assign wr_cons.wid   = cons_wr_if.data.wid;
    assign wr_cons.tbase = cons_wr_if.data.tbase;
    assign wr_cons.slot  = cons_wr_if.data.slot;
    assign wr_cons.mask  = cons_wr_if.data.mask;
    assign wr_cons.data  = cons_wr_if.data.data;
    // Retire-gating: TEX must not retire its handle before the texel commits, or
    // a handle-chained GETW reads a stale slot.
    assign cons_wr_if.ready = gnt_cons;

    // Generic window ops. SETW writes one slot (used in both builds); the
    // GETWF/GETW reads are decoded off `op` in the result mux, and additionally
    // gate is_fastop in a pure-graphics (non-RTU) build.
    wire is_setw  = (op == GFXW_OP_SETW);
    wire is_getwf = (op == GFXW_OP_GETWF);
    wire is_getw  = (op == GFXW_OP_GETW);
    wire is_getws = (op == GFXW_OP_GETWS);
    wire is_read  = is_getwf || is_getw || is_getws;

    // ── result stage (the RAM read is synchronous) ─────────────────────────
    // One op is accepted per cycle into s1; its word arrives from the RAM (or the
    // status file) on the next, where result_if presents it. A pending RAM read
    // still owns the read port's output, so every op that issues a read — and
    // every op that leads to one — must wait for s1 to drain.
    reg                       s1_valid;
    reg                       s1_from_ram;
    sfu_header_t              s1_header;
    reg [NUM_LANES-1:0][31:0] s1_data;

    wire s1_ready = ~s1_valid || result_if.ready;

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

    // Latched terminal status. VX_RT_STS_* fits in a byte, so this is not a
    // full-word file. Scene base is warp-uniform (the CFG uop broadcasts one
    // value to every lane), so it is one word per warp, not per lane.
    reg [7:0]  status  [`VX_CFG_NUM_WARPS][`VX_CFG_NUM_THREADS];
    reg [31:0] rt_scene[`VX_CFG_NUM_WARPS];

    // Async trace bus FSM (shader callbacks). The bus is beat-serial: the ray
    // streams straight out of the window RAM one word per beat, and the hit
    // record streams straight back into it — no ray or hit record is ever
    // registered whole. TRACE2 blocks the warp (at the wait2 PC) until the first
    // response:
    //   TERMINAL — opaque ray finished; write hit window, mark terminal_ready.
    //   CB_YIELD — a non-opaque candidate yielded; write it to the window, raise
    //              the async trap (-> dispatcher), service the dispatcher's
    //              CB_RET as a CB_ACTION, catch the post-resume TERMINAL.
    localparam [3:0] B_IDLE   = 4'd0,
                     B_ARMW   = 4'd1,   // write tmin/tmax into the RAM
                     B_RREAD  = 4'd2,   // set the RAM read addr for a ray beat
                     B_RSEND  = 4'd3,   // drive that ray beat onto the bus
                     B_RSP1   = 4'd4,   // stream the first response into the RAM
                     B_OBJCPY = 4'd5,   // copy the world ray to the object ray
                     B_ARM_WB = 4'd6,   // retire the held arm op (writeback handle)
                     B_CBRET  = 4'd7,   // await the dispatcher's CB_RET op
                     B_CBRD   = 4'd8,   // set the RAM read addr for hit-t
                     B_CBACT  = 4'd9,   // drive the CB_ACTION beat (IS-computed t)
                     B_RSP2   = 4'd10;  // stream the post-resume TERMINAL
    reg [3:0] bstate;

    // Shared beat counter for the streaming states.
    reg [RTU_BEAT_BITS-1:0] fsm_cnt;

    localparam OBJ_WORDS = 6;   // object-ray origin + direction

    // The object-ray copy walks one contiguous span, so origin and direction must
    // abut in both the source and destination slot ranges.
    `STATIC_ASSERT((`VX_RT_RAY_DIRECTION == `VX_RT_RAY_ORIGIN + 3),
        ("ray origin/direction slots must be contiguous"))
    `STATIC_ASSERT((`VX_RT_OBJECT_RAY_DIRECTION == `VX_RT_OBJECT_RAY_ORIGIN + 3),
        ("object-ray origin/direction slots must be contiguous"))

    // Window slot read for request beat `k` — matches RTU_REQ_BEAT order.
    function automatic [GFXW_SLOT_BITS-1:0] req_slot (input [RTU_BEAT_BITS-1:0] k);
        case (k)
            RTU_BEAT_BITS'(0): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_ORIGIN + 0);
            RTU_BEAT_BITS'(1): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_ORIGIN + 1);
            RTU_BEAT_BITS'(2): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_ORIGIN + 2);
            RTU_BEAT_BITS'(3): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_DIRECTION + 0);
            RTU_BEAT_BITS'(4): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_DIRECTION + 1);
            RTU_BEAT_BITS'(5): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_DIRECTION + 2);
            RTU_BEAT_BITS'(6): req_slot = GFXW_SLOT_BITS'(`VX_RT_T_MIN);
            RTU_BEAT_BITS'(7): req_slot = GFXW_SLOT_BITS'(`VX_RT_T_MAX);
            RTU_BEAT_BITS'(8): req_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_FLAGS);
            default:           req_slot = GFXW_SLOT_BITS'(`VX_RT_CULL_MASK);
        endcase
    endfunction

    // Window slot written for response beat `k` — matches RTU_RSP_BEAT order.
    // Beat 7 of a TERMINAL is the status word, which lands in the status latch,
    // not a slot; every other beat targets a real slot.
    function automatic [GFXW_SLOT_BITS-1:0] rsp_slot (input [RTU_BEAT_BITS-1:0] k);
        case (k)
            RTU_BEAT_BITS'(0): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_T);
            RTU_BEAT_BITS'(1): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_BARY_U);
            RTU_BEAT_BITS'(2): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_BARY_V);
            RTU_BEAT_BITS'(3): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_PRIMITIVE_ID);
            RTU_BEAT_BITS'(4): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_INSTANCE_ID);
            RTU_BEAT_BITS'(5): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_GEOMETRY_INDEX);
            RTU_BEAT_BITS'(6): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_INSTANCE_CUSTOM);
            RTU_BEAT_BITS'(7): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_CB_TYPE);
            RTU_BEAT_BITS'(8): rsp_slot = GFXW_SLOT_BITS'(`VX_RT_HIT_SBT_IDX);
            default:           rsp_slot = GFXW_SLOT_BITS'(`VX_RT_CB_HANDLE);
        endcase
    endfunction

    // In-flight trace context (latched at arm) + callback bookkeeping.
    reg [NW_WIDTH-1:0]      if_wid;
    reg [THREAD_BITS-1:0]   if_tbase;
    reg [NUM_LANES-1:0]     if_tmask;
    reg                     yield_owed;     // first rsp was CB_YIELD
    reg [NUM_LANES-1:0]     cb_mask;        // yielding lanes
    reg [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action_lat;
    // Per-warp "terminal landed, wait2 may complete" flag.
    reg [`VX_CFG_NUM_WARPS-1:0] terminal_ready;

    // Register the outgoing RTU bus at this module boundary so the socket->
    // cluster seam launches at a flop (see VX_rtu_bus_slice). The FSM drives the
    // internal working copy; the RTU core registers the response direction.
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (RTU_TAG_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (RTU_TAG_WIDTH),
        .REQ_OUT_BUF (3),  // register our outgoing request
        .RSP_OUT_BUF (0)   // response registered by the RTU core output
    ) rtu_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (rtu_bus_w),
        .bus_out_if (rtu_bus_if)
    );

    // ── request driver (beat-serial) ──────────────────────────────────────
    // TRACE streams RTU_REQ_BEATS ray words straight from the RAM read port;
    // CB_ACTION is a single beat carrying the IS-computed HIT_T (also read from
    // the RAM, held on the port through B_CBACT). scene_base and the per-lane
    // action ride sideband.
    wire in_cbact  = (bstate == B_CBACT);
    wire in_rsend  = (bstate == B_RSEND);
    assign rtu_bus_w.req_valid       = in_rsend || in_cbact;
    assign rtu_bus_w.req_data.kind   = in_cbact ? RTU_REQ_CBACT : RTU_REQ_TRACE;
    assign rtu_bus_w.req_data.eop    = in_cbact ? 1'b1
                                     : (fsm_cnt == RTU_BEAT_BITS'(RTU_REQ_BEATS - 1));
    assign rtu_bus_w.req_data.mask   = in_cbact ? cb_mask : if_tmask;
    assign rtu_bus_w.req_data.data   = core_rdata;             // ray beat / HIT_T
    assign rtu_bus_w.req_data.cb_action  = cb_action_lat;      // CB_ACTION sideband
    assign rtu_bus_w.req_data.scene_base = rt_scene[if_wid];   // TRACE sideband
    assign rtu_bus_w.req_data.tag  = ($bits(rtu_bus_w.req_data.tag))'(execute_if.data.header.uuid);
    `UNUSED_VAR (rtu_bus_w.rsp_data.tag)

    // ── response sink (beat-serial) ───────────────────────────────────────
    // Each beat is one word for a slot (or, on a TERMINAL's status beat, the
    // status latch). The beat is accepted only when its RAM write is granted.
    wire is_yield_rsp   = (rtu_bus_w.rsp_data.kind == RTU_RSP_CBYIELD);
    wire in_rsp         = (bstate == B_RSP1) || (bstate == B_RSP2);
    wire rsp_status_beat = ~is_yield_rsp && (fsm_cnt == RTU_BEAT_BITS'(RTU_RSP_HIT_BEATS));
    // The status beat lands in a latch (no write port), so it needs no grant.
    assign rtu_bus_w.rsp_ready = in_rsp && (rsp_status_beat || gnt_fsm);

    // ── op classification (callback additions) ─────────────────────────────
    wire is_wait   = (op == GFXW_OP_WAIT2);
    wire is_cbret  = (op == GFXW_OP_CB_RET);
    wire is_fastop = ~is_arm && ~is_wait && ~is_cbret; // SETW/CFG/ORIGIN/DIR/GETWF/GETW

    // The held arm op owns execute_if from arm through writeback; in every other
    // state (B_IDLE and the in-trap B_CBRET/B_CBRD/B_CBACT/B_RSP2) execute_if is
    // free for fast window ops — the callback dispatcher reads its payload
    // (GET/GETWF/GETW) before issuing cb_ret. Never lock this PE across a
    // macro-op: the dispatcher must make progress while a trace is parked.
    wire arm_busy = (bstate == B_ARMW) || (bstate == B_RREAD) || (bstate == B_RSEND)
                 || (bstate == B_RSP1) || (bstate == B_OBJCPY) || (bstate == B_ARM_WB);
    // s1_ready is required even though arm writes nothing to the result stage: a
    // pending read still sources its word from the RAM output, and the ray-send
    // walk this arm leads to would re-drive the read port underneath it.
    wire arm_go   = (bstate == B_IDLE) && execute_if.valid && is_arm && s1_ready;

    // Latch the per-lane active mask of the in-flight trace.
    wire [NUM_LANES-1:0] arm_lanes = execute_if.data.header.tmask[thread_base +: NUM_LANES];
`else
    `UNUSED_PARAM (RTU_TAG_WIDTH)
    wire is_fastop = is_setw || is_read;
    wire arm_busy  = 1'b0;
`endif

    // ── execute-side fill writes (SETW / CFG / ORIGIN / DIR) ───────────────
    // One slot per cycle onto the shared write port; the op is held until its
    // last word is granted.
    reg [1:0] fw_cnt;

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [1:0] fw_need = is_setw ? 2'd1 : ((is_cfg || is_origin || is_dir) ? 2'd3 : 2'd0);
`else
    wire [1:0] fw_need = is_setw ? 2'd1 : 2'd0;
`endif

    wire fast_go   = ~arm_busy && execute_if.valid && is_fastop;
    wire fw_more   = fast_go && (fw_cnt < fw_need);
    // Done once every word is placed. The second term retires on the same cycle
    // as the last grant; the first covers a result-stage stall after that grant,
    // where the counter has already advanced past the last word.
    wire fw_done   = (fw_cnt == fw_need)
                  || ((fw_need != 2'd0) && gnt_fill && (fw_cnt == (fw_need - 2'd1)));
    wire fast_fire = fast_go && fw_done && s1_ready;

    // Fill word select.
    reg [GFXW_SLOT_BITS-1:0]  fill_slot;
    reg [NUM_LANES-1:0][31:0] fill_data;
    always @(*) begin
        fill_slot = slot;
        for (integer i = 0; i < NUM_LANES; ++i) begin
            fill_data[i] = execute_if.data.rs1_data[i][31:0];
        end
`ifdef VX_CFG_EXT_RTU_ENABLE
        if (is_cfg) begin
            case (fw_cnt)
                2'd0: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_PAYLOAD_PTR_LO);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = execute_if.data.rs1_data[CFG_L2][31:0];
                    end
                end
                2'd1: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_RAY_FLAGS);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = {16'd0, execute_if.data.rs1_data[CFG_L3][15:0]};
                    end
                end
                default: begin
                    fill_slot = GFXW_SLOT_BITS'(`VX_RT_CULL_MASK);
                    for (integer i = 0; i < NUM_LANES; ++i) begin
                        fill_data[i] = {16'd0, execute_if.data.rs1_data[CFG_L3][31:16]};
                    end
                end
            endcase
        end else if (is_origin || is_dir) begin
            fill_slot = GFXW_SLOT_BITS'((is_origin ? `VX_RT_RAY_ORIGIN : `VX_RT_RAY_DIRECTION) + 32'(fw_cnt));
            for (integer i = 0; i < NUM_LANES; ++i) begin
                case (fw_cnt)
                    2'd0:    fill_data[i] = execute_if.data.rs1_data[i][31:0];
                    2'd1:    fill_data[i] = execute_if.data.rs2_data[i][31:0];
                    default: fill_data[i] = execute_if.data.rs3_data[i][31:0];
                endcase
            end
        end
`endif
    end

    assign req_fill      = fw_more;
    assign wr_fill.wid   = wid;
    assign wr_fill.tbase = thread_base;
    assign wr_fill.slot  = fill_slot;
    assign wr_fill.mask  = execute_if.data.header.tmask[thread_base +: NUM_LANES];
    assign wr_fill.data  = fill_data;

    // ── core read port address select ──────────────────────────────────────
    // GETWS reads a slot-keyed frag record: the block index is warp-uniform, so
    // it is taken from lane 0 rather than decoded per lane.
    wire [NW_WIDTH-1:0] frag_widx = execute_if.data.rs1_data[0][NW_WIDTH-1:0];
    wire [NW_WIDTH-1:0] rd_wid    = is_getws ? frag_widx : wid;

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire cbret_go   = (bstate == B_CBRET) && execute_if.valid && is_cbret;
    wire cbret_fire = cbret_go && s1_ready;
    wire wait_go    = execute_if.valid && is_wait && terminal_ready[wid];
    wire wait_fire  = wait_go && s1_ready;
    wire armwb_fire = (bstate == B_ARM_WB) && execute_if.valid && s1_ready;

    // Ray-beat reads: B_RREAD sets the address, B_RSEND holds it while the beat
    // is presented on the bus. CB_ACTION reads HIT_T across B_CBRD/B_CBACT.
    wire req_rd  = (bstate == B_RREAD) || (bstate == B_RSEND);
    wire cbact_rd = (bstate == B_CBRD) || (bstate == B_CBACT);
    wire obj_rd  = (bstate == B_OBJCPY) && ((fsm_cnt == '0) || (gnt_fsm && (fsm_cnt < RTU_BEAT_BITS'(OBJ_WORDS))));

    assign core_rden  = (fast_fire && is_read) || req_rd || cbact_rd || obj_rd;
    assign core_raddr = req_rd   ? win_addr(if_wid, if_tbase, req_slot(fsm_cnt))
                      : cbact_rd ? win_addr(if_wid, if_tbase, GFXW_SLOT_BITS'(`VX_RT_HIT_T))
                      : obj_rd   ? win_addr(if_wid, if_tbase, GFXW_SLOT_BITS'(`VX_RT_RAY_ORIGIN + 32'(fsm_cnt)))
                                 : win_addr(rd_wid, thread_base, slot);
`else
    assign core_rden  = fast_fire && is_read;
    assign core_raddr = win_addr(rd_wid, thread_base, slot);
`endif

    // ── FSM write source ───────────────────────────────────────────────────
`ifdef VX_CFG_EXT_RTU_ENABLE
    reg [GFXW_SLOT_BITS-1:0]  fsm_slot;
    reg [NUM_LANES-1:0][31:0] fsm_data;
    reg [NUM_LANES-1:0]       fsm_mask;
    reg [NW_WIDTH-1:0]        fsm_wid;
    reg [THREAD_BITS-1:0]     fsm_tbase;
    reg                       fsm_req;
    always @(*) begin
        fsm_req   = 1'b0;
        fsm_wid   = if_wid;
        fsm_tbase = if_tbase;
        fsm_mask  = if_tmask;
        fsm_slot  = '0;
        fsm_data  = '0;
        case (bstate)
            B_ARMW: begin
                // tmin/tmax come from the held ARM uop's operands.
                fsm_req   = 1'b1;
                fsm_wid   = wid;
                fsm_tbase = thread_base;
                fsm_mask  = arm_lanes;
                fsm_slot  = (fsm_cnt == 4'd0) ? GFXW_SLOT_BITS'(`VX_RT_T_MIN)
                                              : GFXW_SLOT_BITS'(`VX_RT_T_MAX);
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    fsm_data[i] = (fsm_cnt == 4'd0) ? execute_if.data.rs1_data[i][31:0]
                                                    : execute_if.data.rs2_data[i][31:0];
                end
            end
            B_OBJCPY: begin
                // Single-level (no TLAS): the object ray equals the world ray, so
                // it is copied within the window rather than widened onto the bus.
                fsm_req  = (fsm_cnt != '0);
                fsm_mask = cb_mask;
                fsm_slot = GFXW_SLOT_BITS'(`VX_RT_OBJECT_RAY_ORIGIN + 32'(fsm_cnt) - 32'd1);
                fsm_data = core_rdata;
            end
            B_RSP1, B_RSP2: begin
                // One response beat -> one slot. The status beat writes the latch,
                // not a slot, so it raises no write request here.
                fsm_req  = rtu_bus_w.rsp_valid && ~rsp_status_beat;
                fsm_mask = is_yield_rsp ? rtu_bus_w.rsp_data.cb_active_mask[NUM_LANES-1:0] : if_tmask;
                fsm_slot = rsp_slot(fsm_cnt);
                fsm_data = rtu_bus_w.rsp_data.data;
            end
            default:;
        endcase
    end

    assign req_fsm      = fsm_req;
    assign wr_fsm.wid   = fsm_wid;
    assign wr_fsm.tbase = fsm_tbase;
    assign wr_fsm.slot  = fsm_slot;
    assign wr_fsm.mask  = fsm_mask;
    assign wr_fsm.data  = fsm_data;
`else
    assign req_fsm  = 1'b0;
    assign wr_fsm   = '0;
`endif

    // ── async trap raise on CB_YIELD ───────────────────────────────────────
    // Fires as the held arm op retires, so the warp — still parked at the wait2
    // PC — is redirected to the dispatcher with mepc = wait2 PC.
`ifdef VX_CFG_EXT_RTU_ENABLE
    assign async_trap_if.valid  = armwb_fire && yield_owed;   // trap entry: callback yield
    assign async_trap_if.unlock = armwb_fire;                 // resume the wstall'd trace warp
    assign async_trap_if.wid    = if_wid;
    assign async_trap_if.cause  = `VX_CFG_XLEN'(`VX_TRAP_CAUSE_RTU_CALLBACK);
    assign async_trap_if.tmask  = (`VX_CFG_NUM_THREADS'(cb_mask)) << if_tbase;
`endif

    // ── sequential state ───────────────────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin
            s1_valid <= 1'b0;
            fw_cnt   <= 2'd0;
`ifdef VX_CFG_EXT_RTU_ENABLE
            bstate         <= B_IDLE;
            fsm_cnt        <= 4'd0;
            yield_owed     <= 1'b0;
            terminal_ready <= '0;
`endif
        end else begin
            // result stage
            if (s1_valid && result_if.ready) begin
                s1_valid <= 1'b0;
            end
            if (fast_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= is_read;
                s1_header   <= execute_if.data.header;
                s1_data     <= '0;
            end

            // fill word advance
            if (fast_fire) begin
                fw_cnt <= 2'd0;
            end else if (gnt_fill) begin
                fw_cnt <= fw_cnt + 2'd1;
            end

`ifdef VX_CFG_EXT_RTU_ENABLE
            if (wait_fire || cbret_fire || armwb_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= 1'b0;
                s1_header   <= execute_if.data.header;
                s1_data     <= '0;
            end
            if (wait_fire) begin
                // WAIT2 returns the latched terminal status (byte-wide, zero-extended).
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    s1_data[i] <= 32'(status[wid][thread_base + THREAD_BITS'(i)]);
                end
                terminal_ready[wid] <= 1'b0;
            end
            if (cbret_fire) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    cb_action_lat[i] <= execute_if.data.rs1_data[i][RTU_CB_ACTION_BITS-1:0];
                end
            end

            // ── bus FSM ────────────────────────────────────────────────
            case (bstate)
            B_IDLE: begin
                        if (arm_go) begin
                            if_wid   <= wid;
                            if_tbase <= thread_base;
                            if_tmask <= arm_lanes;
                            fsm_cnt  <= '0;
                            bstate   <= B_ARMW;
                        end
                    end
            B_ARMW: begin
                        // Write tmin then tmax into the RAM so a later GETW sees
                        // them; the ray request then reads all 10 words back.
                        if (gnt_fsm) begin
                            if (fsm_cnt == RTU_BEAT_BITS'(1)) begin
                                fsm_cnt <= '0;
                                bstate  <= B_RREAD;
                            end else begin
                                fsm_cnt <= fsm_cnt + RTU_BEAT_BITS'(1);
                            end
                        end
                    end
            B_RREAD: bstate <= B_RSEND;   // RAM read issued; word ready next cycle
            B_RSEND: begin
                        if (rtu_bus_w.req_ready) begin
                            if (fsm_cnt == RTU_BEAT_BITS'(RTU_REQ_BEATS - 1)) begin
                                fsm_cnt <= '0;
                                bstate  <= B_RSP1;
                            end else begin
                                fsm_cnt <= fsm_cnt + RTU_BEAT_BITS'(1);
                                bstate  <= B_RREAD;
                            end
                        end
                    end
            B_RSP1: begin
                        if (rtu_bus_w.rsp_valid && rtu_bus_w.rsp_ready) begin
                            if (rsp_status_beat) begin
                                for (integer i = 0; i < NUM_LANES; ++i) begin
                                    if (if_tmask[i]) begin
                                        status[if_wid][if_tbase + THREAD_BITS'(i)] <= rtu_bus_w.rsp_data.data[i][7:0];
                                    end
                                end
                            end
                            if (rtu_bus_w.rsp_data.eop) begin
                                fsm_cnt <= '0;
                                if (is_yield_rsp) begin
                                    cb_mask    <= rtu_bus_w.rsp_data.cb_active_mask[NUM_LANES-1:0];
                                    yield_owed <= 1'b1;
                                    bstate     <= B_OBJCPY;   // a yield owes the object ray
                                end else begin
                                    terminal_ready[if_wid] <= 1'b1;
                                    yield_owed <= 1'b0;
                                    bstate     <= B_ARM_WB;
                                end
                            end else begin
                                fsm_cnt <= fsm_cnt + RTU_BEAT_BITS'(1);
                            end
                        end
                    end
            B_OBJCPY: begin
                        // Step 0 only reads; step k writes word k-1 and reads word k.
                        if (fsm_cnt == '0) begin
                            fsm_cnt <= RTU_BEAT_BITS'(1);
                        end else if (gnt_fsm) begin
                            if (fsm_cnt == RTU_BEAT_BITS'(OBJ_WORDS)) begin
                                bstate <= B_ARM_WB;
                            end else begin
                                fsm_cnt <= fsm_cnt + RTU_BEAT_BITS'(1);
                            end
                        end
                    end
            B_ARM_WB: begin
                        if (armwb_fire) begin
                            // handle writeback retires; async trap fired this cycle if yield.
                            yield_owed <= 1'b0;
                            bstate     <= yield_owed ? B_CBRET : B_IDLE;
                        end
                    end
            B_CBRET: begin
                        if (cbret_fire) begin
                            bstate <= B_CBRD;
                        end
                    end
            B_CBRD: bstate <= B_CBACT;   // HIT_T read issued; held on the port
            B_CBACT: begin
                        if (rtu_bus_w.req_ready) begin
                            fsm_cnt <= '0;
                            bstate  <= B_RSP2;
                        end
                    end
            B_RSP2: begin
                        if (rtu_bus_w.rsp_valid && rtu_bus_w.rsp_ready) begin
                            if (rsp_status_beat) begin
                                for (integer i = 0; i < NUM_LANES; ++i) begin
                                    if (if_tmask[i]) begin
                                        status[if_wid][if_tbase + THREAD_BITS'(i)] <= rtu_bus_w.rsp_data.data[i][7:0];
                                    end
                                end
                            end
                            if (rtu_bus_w.rsp_data.eop) begin
                                terminal_ready[if_wid] <= 1'b1;
                                fsm_cnt <= '0;
                                bstate  <= B_IDLE;
                            end else begin
                                fsm_cnt <= fsm_cnt + RTU_BEAT_BITS'(1);
                            end
                        end
                    end
            default:;
            endcase

            // Scene base is warp-uniform; the CFG uop broadcasts one value.
            if (fast_go && is_cfg) begin
                rt_scene[wid] <= execute_if.data.rs1_data[CFG_L1][31:0];
            end
`endif
        end
    end

    // ── result path ───────────────────────────────────────────────────────
    sfu_result_t rsp_data_out;
    assign rsp_data_out.header = s1_header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        wire [31:0] rdata = s1_from_ram ? core_rdata[i] : s1_data[i];
        assign rsp_data_out.data[i] = `VX_CFG_XLEN'(rdata);
    end

    assign result_if.valid = s1_valid;
    assign result_if.data  = rsp_data_out;

`ifdef VX_CFG_EXT_RTU_ENABLE
    assign execute_if.ready = fast_fire || wait_fire || cbret_fire || armwb_fire;
`else
    assign execute_if.ready = fast_fire;
`endif

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
