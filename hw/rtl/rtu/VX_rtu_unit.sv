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

    // TRACE wstall release + trace-gate reopen (-> scheduler / scoreboard)
    VX_sched_unlock_if.master sched_unlock_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam RAM_SIZE  = `VX_CFG_NUM_WARPS * GFXW_SLOT_COUNT;
    localparam RAM_DATAW = 32 * NUM_LANES;

    // The address concatenates {warp, slot}; the slot stride must be a power of
    // two or that concatenation degenerates into a multiplier.
    `STATIC_ASSERT(((1 << GFXW_SLOT_BITS) == GFXW_SLOT_COUNT),
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

    wire [GFXW_OP_BITS-1:0]   op   = execute_if.data.op_args.gfxw.op;
    wire [2:0]                uop  = execute_if.data.op_args.gfxw.uop;
    wire [GFXW_SLOT_BITS-1:0] slot = execute_if.data.op_args.gfxw.slot[GFXW_SLOT_BITS-1:0];
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
        .ARM_OUT_BUF (0),  // MUST be 0: the arm's ready IS the RTU's readiness
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
    wire [GFXW_SLOT_BITS-1:0] win_slot = rtu_bus_w.win_data.slot;
    `UNUSED_VAR (rtu_bus_w.win_data.tag)

    assign rtu_bus_w.win_ready = 1'b1;

    wire                       core_rden;
    wire [GFXW_SLOT_BITS-1:0]  core_slot;
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
        .raddr ({wid, core_slot}),
        .rdata (core_rdata)
    );

    // ── op classification ──────────────────────────────────────────────────
    wire is_trace = (op == GFXW_OP_TRACE);
    wire is_cfg   = is_trace && (uop == GFXW_UOP_CFG);   // rings the arm
    wire is_ray   = is_trace && ~is_cfg;                 // ORIGIN | DIR | ARM
    wire is_wait  = (op == GFXW_OP_WAIT);
    wire is_cont  = (op == GFXW_OP_CB_RET);              // CONTINUE reuses CB_RET

    // Ops whose result word comes out of the RAM. WAIT is one of them: the status
    // it returns IS a slot, written by the RTU as the last beat of a record.
    wire is_read = (op == GFXW_OP_GETWF) || (op == GFXW_OP_GETW) || is_wait;

    reg [`VX_CFG_NUM_WARPS-1:0] response_ready, trace_open;

    // ── result stage (the RAM read is synchronous) ─────────────────────────
    // One op is accepted per cycle into s1; its word arrives from the RAM on the
    // next, where result_if presents it.
    reg          s1_valid;
    reg          s1_from_ram;
    sfu_header_t s1_header;

    wire s1_ready = ~s1_valid || result_if.ready;

    // ── the arm doorbell: the CFG uop ──────────────────────────────────────
    // Lane-packed config rides lanes 1..3 of rs1 (the implicit vx_wgather layout:
    // lane1=scene, lane2=payload, lane3=flags|cull). The trace ABI requires
    // NUM_LANES >= 4; clamp the indices so narrower builds (which never issue
    // TRACE) still elaborate.
    localparam CFG_L1 = (NUM_LANES > 1) ? 1 : 0;
    localparam CFG_L2 = (NUM_LANES > 2) ? 2 : 0;
    localparam CFG_L3 = (NUM_LANES > 3) ? 3 : 0;

    // The one point in a TRACE that can block, and the reason the arm channel is
    // unbuffered end to end: this uop does not retire until the RTU has actually
    // taken the ray, so the RAY beats behind it can never reach a traversal that
    // is not theirs. TRACE is not issued unless this core's RTU is free (the trace
    // gate in VX_issue), so the block is only ever another CORE contending for a
    // shared RTU — never a warp that is itself waiting on this core.
    assign rtu_bus_w.arm_valid            = execute_if.valid && is_cfg && s1_ready;
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
    wire is_2beat  = cont_want || (is_trace && (uop == GFXW_UOP_ARM));

    reg  [1:0] rb_cnt;
    wire [1:0] rb_last = is_2beat ? 2'd1 : 2'd2;   // index of the final beat

    // Hold the LAST beat until the result stage can take the op with it: then a
    // beat leaving is the op retiring, and there is no "beats done, op still here"
    // state to track.
    wire rb_end  = (rb_cnt == rb_last);
    wire rb_more = execute_if.valid && is_stream && (~rb_end || s1_ready);
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
    assign rtu_bus_w.req_data.data      = rb_data;
    assign rtu_bus_w.req_data.cb_action = cont_act;

    // ── the status write ──────────────────────────────────────────────────
    // The last write of a record: it releases the parked WAIT and reopens or
    // closes the trace. A TERMINAL record also frees the RTU, which reopens the
    // scoreboard's trace gate.
    wire status_wr = win_v && (win_slot == GFXW_SLOT_BITS'(`VX_RT_STATUS));

    assign sched_unlock_if.trace_done = status_wr && ~rtu_bus_w.win_data.is_cand;

    // ── op completion ─────────────────────────────────────────────────────
    // A stream op retires with its last beat; everything else needs only the
    // result stage. WAIT blocks until a record has landed.
    wire read_fire   = execute_if.valid && is_read && (~is_wait || response_ready[wid]) && s1_ready;
    wire stream_fire = rb_fire && rb_end;
    wire cont_nop    = execute_if.valid && is_cont && ~trace_open[wid] && s1_ready;
    wire op_fire     = read_fire || stream_fire || cont_nop || arm_fire;

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
    wire trace_burst_end = stream_fire && is_trace && (uop == GFXW_UOP_ARM);
    assign sched_unlock_if.valid = trace_burst_end || (read_fire && is_wait);
    assign sched_unlock_if.wid   = wid;

    assign core_rden = read_fire;
    assign core_slot = is_wait ? GFXW_SLOT_BITS'(`VX_RT_STATUS) : slot;

    // ── sequential state ───────────────────────────────────────────────────
    always @(posedge clk) begin
        if (reset) begin
            s1_valid       <= 1'b0;
            rb_cnt         <= 2'd0;
            response_ready <= '0;
            trace_open     <= '0;
        end else begin
            if (s1_valid && result_if.ready) begin
                s1_valid <= 1'b0;
            end
            if (op_fire) begin
                s1_valid    <= 1'b1;
                s1_from_ram <= is_read;
                s1_header   <= execute_if.data.header;
            end

            // The last beat retires the op, so the counter always lands back at 0.
            if (rb_fire) begin
                rb_cnt <= rb_end ? 2'd0 : (rb_cnt + 2'd1);
            end

            // WAIT consumes the record so the next one blocks again.
            if (read_fire && is_wait) begin
                response_ready[wid] <= 1'b0;
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
