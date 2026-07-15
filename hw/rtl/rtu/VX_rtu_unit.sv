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

// VX_rtu_unit — the RTU's per-core SFU PE. It is a slot RAM, a beat streamer,
// and three bits per warp.
//
// The RAM is the hit window, and it holds traversal RESULTS only:
//   the RTU is its ONLY writer    — no instruction writes a slot
//   the shader is its ONLY reader — GETWF/GETW/WAIT read a slot into the FP/GP file
//   the RTU never reads it back   — so it has no read port facing the RTU
// One writer, one reader: a plain 1W1R BRAM, no arbiter, no mirror. It is
// lane-packed (one word holds every lane's copy of a slot) and addressed by
// {warp, slot} — the RTU traces a whole warp, so there is no simd group to index.
//
// A ray therefore never lands here. The TRACE burst hands it to the traversal
// datapath directly (see VX_rtu_bus_if): the CFG uop rings the arm doorbell with
// the ray's warp-uniform half, and ORIGIN/DIR/ARM stream the per-lane half out of
// their own register operands as RAY beats. Nothing buffers it on the way — the
// RTU core's per-context ray state IS the destination. An intersection shader's t
// and hitAttribute come back the same way, on its CONTINUE.
//
// The two bits, both set by the RTU's write to the status slot, which is by
// construction the last write of a record:
//   response_ready — a record landed; the blocked WAIT may complete
//   trace_open     — that record was a candidate; a CONTINUE may resume it

`include "VX_define.vh"

module VX_rtu_unit import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter RTU_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // socket-shared RTU bus (the RTU core is the master; this unit is its slave)
    VX_rtu_bus_if.master    rtu_bus_if,

    // warp unlock (-> scheduler)
    VX_sched_unlock_if.master sched_unlock_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam RAM_SIZE  = `VX_CFG_NUM_WARPS * RTUW_SLOT_COUNT;
    localparam RAM_DATAW = 32 * NUM_LANES;

    // The address concatenates {warp, slot}; the slot stride must be a power of
    // two or that concatenation degenerates into a multiplier.
    `STATIC_ASSERT(((1 << RTUW_SLOT_BITS) == RTUW_SLOT_COUNT),
        ("window slot count must be a power of two"))

    // The RTU traces a whole warp at a time: one arm hands it one ray per lane,
    // and it holds exactly one. A narrower SFU would split a TRACE into a simd
    // group per doorbell, and the second group's arm — which the RTU cannot accept
    // until the first group's walk is done — would sit at the head of the SFU with
    // the first group's ray beats queued behind it, deadlocking the walk it is
    // waiting on. Trace a full warp, or do not build the RTU. (This is also what
    // makes the RAM address a bare {warp, slot}.)
    `STATIC_ASSERT((NUM_LANES == `VX_CFG_NUM_THREADS),
        ("the RTU requires VX_CFG_NUM_SFU_LANES == VX_CFG_NUM_THREADS"))

    wire [RTUW_OP_BITS-1:0]   op   = execute_if.data.op_args.rtuw.op;
    wire [2:0]                uop  = execute_if.data.op_args.rtuw.uop;
    wire [RTUW_SLOT_BITS-1:0] slot = execute_if.data.op_args.rtuw.slot[RTUW_SLOT_BITS-1:0];
    wire [NW_WIDTH-1:0]       wid  = execute_if.data.header.wid;

    // Register the outgoing RTU bus at this module boundary so the core->socket
    // seam launches at a flop (see VX_rtu_bus_slice). We drive the internal
    // working copy; the RTU core registers the direction it sources.
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (RTU_TAG_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (RTU_TAG_WIDTH),
        .SRC_WIDTH   (1),  // a unit does not know its own index; the arbiter fills it in
        .ARM_OUT_BUF (3),  // the RTU always takes an arm, so this may be registered
        .REQ_OUT_BUF (3),  // register the ray/continue beats we source
        .SLV_OUT_BUF (0)   // win registered by the RTU core output
    ) rtu_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (rtu_bus_w),
        .bus_out_if (rtu_bus_if)
    );

    // ── the slot RAM ───────────────────────────────────────────────────────
    // Written by the RTU, read by the shader. Neither port is ever contended, so
    // the RTU is never made to wait for a write and a read never waits for it.
    wire                      win_v    = rtu_bus_w.win_valid;
    wire [NW_WIDTH-1:0]       win_wid  = rtu_bus_w.win_data.wid;
    wire [RTUW_SLOT_BITS-1:0] win_slot = rtu_bus_w.win_data.slot;
    `UNUSED_VAR (rtu_bus_w.win_data.tag)

    assign rtu_bus_w.win_ready = 1'b1;

    wire                       core_rden;
    wire [RTUW_SLOT_BITS-1:0]  core_slot;
    wire [NUM_LANES-1:0][31:0] core_rdata;

    VX_dp_ram #(
        .DATAW    (RAM_DATAW),
        .SIZE     (RAM_SIZE),
        .WRENW    (NUM_LANES),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) win_ram (
        .clk   (clk),
        .reset (reset),
        .read  (core_rden),
        .write (win_v),
        .wren  (rtu_bus_w.win_data.mask),
        .waddr ({win_wid, win_slot}),
        .wdata (rtu_bus_w.win_data.data),
        .raddr ({core_wid, core_slot}),
        .rdata (core_rdata)
    );

    // ── op classification ──────────────────────────────────────────────────
    wire is_trace = (op == RTUW_OP_TRACE);
    wire is_cfg   = is_trace && (uop == RTUW_UOP_CFG);   // rings the arm
    wire is_ray   = is_trace && ~is_cfg;                 // ORIGIN | DIR | ARM
    wire is_wait  = (op == RTUW_OP_WAIT);
    wire is_cont  = (op == RTUW_OP_CB_RET);              // CONTINUE reuses CB_RET

    // Ops whose result word comes out of the RAM, AT THE EXECUTE STAGE. WAIT is no
    // longer one of them — see the parked-WAIT table below.
    wire is_read = (op == RTUW_OP_GETWF) || (op == RTUW_OP_GETW);

    reg [`VX_CFG_NUM_WARPS-1:0] response_ready, trace_open;

    // ── the parked WAIT — a long-latency op MUST NOT hold the execute stage ────
    // A WAIT used to sit at execute_if until its record landed, and execute_if is
    // ONE in-order port shared by every warp. So a warp waiting on the RTU
    // head-of-line blocked every OTHER warp's TRACE burst — the very ops that would
    // have fed the RTU its next ray. The RTU then starved waiting for work that was
    // queued behind a warp waiting for the RTU.
    //
    // Measured on rt_raycast: the RTU sat idle with nothing to run for 44% of all
    // cycles, while the time it spent idle with a ray already in hand was 117 cycles
    // in the whole program. Starvation is the cost here, not ray-delivery latency.
    //
    // So a WAIT RETIRES from execute_if immediately and parks here. Its writeback is
    // issued when the record lands. The warp does not run on — decode wstalls it at
    // WAIT and the unlock below is what releases it — so the semantics are unchanged;
    // only the execute port is freed. A warp holds at most one trace, so one entry
    // per warp is exact, not a heuristic.
    reg [`VX_CFG_NUM_WARPS-1:0] wait_pend;
    sfu_header_t                wait_hdr [`VX_CFG_NUM_WARPS];

    // ── result stage (the RAM read is synchronous) ─────────────────────────
    // One op is accepted per cycle into s1; its word arrives from the RAM on the
    // next, where result_if presents it.
    reg          s1_valid;
    reg          s1_from_ram;
    sfu_header_t s1_header;

    wire s1_ready = ~s1_valid || result_if.ready;

    // A waking parked WAIT (below) OWNS the result stage in the cycle it fires. Every
    // execute-stage op that needs a writeback must therefore hold off, or it would
    // retire from execute_if with its result silently dropped — and its scoreboard
    // entry would never clear. Only the WAIT *park* itself may proceed alongside a
    // wake: parking produces no word.
    wire wake_fire;
    wire s1_grant = s1_ready && ~wake_fire;

    // ── the arm doorbell: the CFG uop ──────────────────────────────────────
    // Lane-packed config rides lanes 1..3 of rs1 (the implicit vx_wgather layout:
    // lane1=scene, lane2=payload, lane3=flags|cull). The trace ABI requires
    // NUM_LANES >= 4; clamp the indices so narrower builds (which never issue
    // TRACE) still elaborate.
    localparam CFG_L1 = (NUM_LANES > 1) ? 1 : 0;
    localparam CFG_L2 = (NUM_LANES > 2) ? 2 : 0;
    localparam CFG_L3 = (NUM_LANES > 3) ? 3 : 0;

    // The arm CANNOT block: the RTU keeps a ray slot per warp, and a warp holds at
    // most one trace (decode wstalls it at TRACE and it does not run past its WAIT),
    // so the slot this arm targets is free by construction. That is what makes a
    // TRACE burst unable to wedge the issue lock -- and therefore what lets the
    // issue stage know nothing about the RTU.
    assign rtu_bus_w.arm_valid            = execute_if.valid && is_cfg && s1_grant;
    assign rtu_bus_w.arm_data.src         = 1'b0;   // the arbiter's grant is the real one
    assign rtu_bus_w.arm_data.wid         = wid;
    assign rtu_bus_w.arm_data.mask        = execute_if.data.header.tmask;
    assign rtu_bus_w.arm_data.scene_base  = execute_if.data.rs1_data[CFG_L1][`VX_CFG_MEM_ADDR_WIDTH-1:0];
    assign rtu_bus_w.arm_data.payload_ptr = execute_if.data.rs1_data[CFG_L2][31:0];
    assign rtu_bus_w.arm_data.flags       = execute_if.data.rs1_data[CFG_L3][15:0];
    assign rtu_bus_w.arm_data.cull_mask   = execute_if.data.rs1_data[CFG_L3][31:16];
    assign rtu_bus_w.arm_data.tag         = ($bits(rtu_bus_w.arm_data.tag))'(execute_if.data.header.uuid);

    wire arm_fire = rtu_bus_w.arm_valid && rtu_bus_w.arm_ready;

    // ── the req channel: RAY beats, and the warp's CONTINUE ────────────────
    // Both stream one per-lane word per beat straight out of the op's registers,
    // so the op is held at execute_if until its last beat is taken:
    //
    //   ORIGIN : 3 beats — rs1..rs3 = f0,f1,f2 (origin)
    //   DIR    : 3 beats — rs1..rs3 = f3,f4,f5 (direction)
    //   ARM    : 2 beats — rs1,rs2  = f6,f7    (t_min, t_max)
    //   CONT   : 2 beats — rs2,rs3  = the shader's t, then its hitAttribute
    //
    // Every beat is one register, so the source is just an index — the ray walks
    // rs1..rs3 and a CONT walks the same list from rs2 (it spends rs1 on the
    // action). The RTU is always ready here (it is waiting for exactly these), so
    // a beat only ever waits on the outgoing register slice.
    //
    // A CONTINUE with no open candidate retires as a no-op (defensive — correct
    // kernels only issue one inside the yield loop).
    wire cont_want = is_cont && trace_open[wid];
    wire is_stream = is_ray || cont_want;
    wire is_2beat  = cont_want || (is_trace && (uop == RTUW_UOP_ARM));

    reg  [1:0] rb_cnt;
    wire [1:0] rb_last = is_2beat ? 2'd1 : 2'd2;   // index of the final beat

    // Hold the LAST beat until the result stage can take the op with it: then a
    // beat leaving is the op retiring, and there is no "beats done, op still here"
    // state to track.
    wire rb_end  = (rb_cnt == rb_last);
    wire rb_more = execute_if.valid && is_stream && (~rb_end || s1_grant);
    wire rb_fire = rb_more && rtu_bus_w.req_ready;

    // Beat source select: rs1,rs2,rs3 by index — shifted up one for a CONT.
    wire [1:0] rb_src = cont_want ? (rb_cnt + 2'd1) : rb_cnt;

    reg [NUM_LANES-1:0][31:0] rb_data;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            case (rb_src)
                2'd0:    rb_data[i] = execute_if.data.rs1_data[i][31:0];
                2'd1:    rb_data[i] = execute_if.data.rs2_data[i][31:0];
                default: rb_data[i] = execute_if.data.rs3_data[i][31:0];
            endcase
        end
    end

    // The CONTINUE carries every active lane's action word, including lanes that
    // are still traversing (PENDING) and whose action is meaningless. The
    // scheduler applies an action only to a lane that actually yielded a
    // candidate, so those ride along harmlessly.
    wire [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_act;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cont_act
        assign cont_act[i] = execute_if.data.rs1_data[i][RTU_CB_ACTION_BITS-1:0];
    end

    assign rtu_bus_w.req_valid          = rb_more;
    assign rtu_bus_w.req_data.kind      = cont_want ? RTU_REQ_CONT : RTU_REQ_RAY;
    assign rtu_bus_w.req_data.src       = 1'b0;  // the arbiter's grant is the real one
    assign rtu_bus_w.req_data.wid       = wid;   // the beat's owner; see VX_rtu_bus_if
    assign rtu_bus_w.req_data.data      = rb_data;
    assign rtu_bus_w.req_data.cb_action = cont_act;

    // ── the status write ──────────────────────────────────────────────────
    // The last write of a record: it unblocks the warp's WAIT, and reopens or
    // closes the trace for a CONTINUE.
    wire status_wr = win_v && (win_slot == RTUW_SLOT_BITS'(`VX_RT_STATUS));

    // ── the parked-WAIT completion: pick a warp whose record has landed ───
    // Its status word comes out of the same RAM port an execute-stage read uses, so
    // the two arbitrate for the result stage. The parked WAIT wins: it has already
    // retired from execute_if and the warp behind it is stalled on nothing else,
    // while a GETW at execute_if can simply wait a cycle.
    wire [`VX_CFG_NUM_WARPS-1:0] wait_done = wait_pend & response_ready;

    wire [NW_WIDTH-1:0] wake_wid;
    wire                wake_valid;
    wire [`VX_CFG_NUM_WARPS-1:0] wake_1h;
    VX_priority_encoder #(
        .N (`VX_CFG_NUM_WARPS)
    ) wake_picker (
        .data_in    (wait_done),
        .onehot_out (wake_1h),
        .index_out  (wake_wid),
        .valid_out  (wake_valid)
    );
    `UNUSED_VAR (wake_1h)

    assign wake_fire = wake_valid && s1_ready;

    // ── op completion ─────────────────────────────────────────────────────
    // A stream op retires with its last beat; everything else needs only the
    // result stage. A WAIT retires the moment it is parked — it never blocks.
    wire read_fire   = execute_if.valid && is_read && s1_grant;
    wire wait_fire   = execute_if.valid && is_wait && ~wait_pend[wid];
    wire stream_fire = rb_fire && rb_end;
    wire cont_nop    = execute_if.valid && is_cont && ~trace_open[wid] && s1_grant;
    wire op_fire     = read_fire || wait_fire || stream_fire || cont_nop || arm_fire;

    `RUNTIME_ASSERT(~(execute_if.valid && is_wait && wait_pend[wid]),
        ("%t: *** %s: wid=%0d issued a second WAIT with one already parked",
            $time, INSTANCE_ID, wid))

    // ── the warp unlock ───────────────────────────────────────────────────
    // Both ops that wstall their warp at decode are released here, by the op
    // itself retiring — never by a traversal event, so the pulse can never race
    // the stall it clears (the stall is set at decode, strictly earlier).
    //
    //   TRACE — the ARM uop's last beat: the RTU now has the ray, so the burst is
    //           done. The warp runs on to its WAIT, which re-stalls it.
    //   WAIT  — read_fire already requires the record (response_ready), so the
    //           status this returns is the one the warp was waiting for.
    //
    // The two are mutually exclusive: one op is at execute_if at a time, and a
    // stream op is never a read.
    wire trace_burst_end = stream_fire && is_trace && (uop == RTUW_UOP_ARM);
    assign sched_unlock_if.valid = trace_burst_end || wake_fire;
    assign sched_unlock_if.wid   = trace_burst_end ? wid : wake_wid;

    // The RAM read: a waking WAIT reads its warp's STATUS slot; an execute-stage
    // read reads the slot its op names.
    assign core_rden = read_fire || wake_fire;
    assign core_slot = wake_fire ? RTUW_SLOT_BITS'(`VX_RT_STATUS) : slot;
    wire [NW_WIDTH-1:0] core_wid = wake_fire ? wake_wid : wid;

    // ── sequential state ───────────────────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin
            s1_valid       <= 1'b0;
            rb_cnt         <= 2'd0;
            response_ready <= '0;
            trace_open     <= '0;
            wait_pend      <= '0;
        end else begin
            if (s1_valid && result_if.ready) begin
                s1_valid <= 1'b0;
            end
            // A waking WAIT owns the result stage; otherwise the execute-stage op does.
            // A parked WAIT does NOT enter s1 when it retires — it has no word yet.
            if (wake_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= 1'b1;
                s1_header   <= wait_hdr[wake_wid];
            end else if (op_fire && ~wait_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= is_read;
                s1_header   <= execute_if.data.header;
            end

            // park the WAIT: it retires now, and answers later
            if (wait_fire) begin
                wait_pend[wid] <= 1'b1;
                wait_hdr[wid]  <= execute_if.data.header;
            end

            // The last beat retires the op, so the counter always lands back at 0.
            if (rb_fire) begin
                rb_cnt <= rb_end ? 2'd0 : (rb_cnt + 2'd1);
            end

            // The waking WAIT consumes the record, so the next one waits again.
            if (wake_fire) begin
                wait_pend[wake_wid]      <= 1'b0;
                response_ready[wake_wid] <= 1'b0;
            end
            // A CONTINUE resolves the open candidate once its last beat is away.
            if (stream_fire && cont_want) begin
                trace_open[wid] <= 1'b0;
            end
            // Last, so a record landing in the same cycle as the op it unblocks
            // wins: the record is the newer event.
            if (status_wr) begin
                response_ready[win_wid] <= 1'b1;
                trace_open[win_wid]     <= rtu_bus_w.win_data.is_cand;
            end
        end
    end

    // ── result path ───────────────────────────────────────────────────────
    // Only a slot read returns a word. CFG's handle and the posted stream ops
    // write back zero.
    sfu_result_t rsp_data_out;
    assign rsp_data_out.header = s1_header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign rsp_data_out.data[i] = `VX_CFG_XLEN'(s1_from_ram ? core_rdata[i] : 32'd0);
    end

    assign result_if.valid  = s1_valid;
    assign result_if.data   = rsp_data_out;
    assign execute_if.ready = op_fire;

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%t: %s rtuw-op: wid=%0d, PC=0x%0h, tmask=%b, op=%0d, slot=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, op, slot, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
