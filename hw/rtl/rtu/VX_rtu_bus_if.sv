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

`include "VX_define.vh"

// VX_rtu_bus_if — per-core RTU unit <-> socket-shared RTU core channel.
//
// The window is RESULT STORAGE, nothing else. The RTU is its only writer, the
// shader its only reader, and the RTU never reads it back: a ray reaches the
// traversal datapath by streaming over this bus from the TRACE burst's own
// register operands, so the window holds no ray, no traversal state, and needs
// no read port facing the RTU.
//
// Three channels, sized so that no channel can head-of-line block another:
//
//   arm  (unit -> RTU)  A warp armed a TRACE, and the ray is about to follow.
//                       Carries the warp's identity plus the warp-uniform part
//                       of the ray (scene_base, flags, cull_mask, payload_ptr),
//                       which would otherwise cost a per-lane beat each.
//                       MAY BLOCK: the RTU accepts an arm only when idle. It is
//                       the ONLY point in a TRACE that can block, and TRACE is
//                       not issued unless this core's RTU can take it (see the
//                       trace gate in VX_scoreboard), so the block is only ever
//                       cross-core contention for a shared RTU — never a wait on
//                       a warp that is itself waiting on this core.
//
//   req  (unit -> RTU)  Everything the RTU is waiting FOR, so the RTU is ALWAYS
//                       ready here and this channel can never stall:
//                         RAY   — one ray word per lane, in arrival order (see
//                                 RTU_RAY_BEATS); the beats that follow an arm
//                         CONT  — the warp's CONTINUE: per-lane action + the
//                                 intersection shader's t and hitAttribute,
//                                 resuming an open candidate
//                       RAY and CONT can share one channel because they are
//                       never outstanding at the same time — a trace's ray is
//                       whole long before it can yield a candidate.
//
//   win  (RTU -> unit)  One masked slot WRITE per beat, each carrying its own
//                       {wid, slot} address. Writes of one response are
//                       an ordered stream, so the status slot — written LAST —
//                       completes the parked WAIT only once the record is whole.
//
// Splitting `arm` out is what makes this deadlock-free. If an arm shared the
// `req` channel, a second warp's arm could sit at the arbiter head while the
// RTU refused it (busy) and the active warp's CONT queued behind it — the RTU
// waiting for a CONT that can never arrive.

interface VX_rtu_bus_if import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();
    // ── arm: a warp armed a TRACE; its ray follows on `req` ───────────
    // The warp-uniform half of the ray rides here: one copy, not NUM_LANES.
    typedef struct packed {
        logic [NW_WIDTH-1:0]                wid;
        logic [NUM_LANES-1:0]               mask;        // active lanes of the trace
        logic [`VX_CFG_MEM_ADDR_WIDTH-1:0]  scene_base;  // device address of the scene
        logic [15:0]                        flags;       // VX_RT_FLAG_*
        logic [15:0]                        cull_mask;
        logic [31:0]                        payload_ptr; // the RTU stages it for the callbacks
        logic [TAG_WIDTH-1:0]               tag;
    } arm_data_t;

    // ── req: RAY beats | the warp's CONTINUE ──────────────────────────
    // `data` is the one per-lane word field, reused by both kinds: a ray word for
    // RAY, and for CONT the shader's t (beat 0) then its hitAttribute (beat 1).
    // A two-beat CONT costs a cycle and no wires; a second per-lane field would
    // cost 32*NUM_LANES of them.
    typedef struct packed {
        logic                                         kind;      // RTU_REQ_*
        logic [NUM_LANES-1:0][31:0]                   data;
        logic [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action; // CONT beat 0
    } req_data_t;

    // ── win: one masked window slot WRITE ─────────────────────────────
    typedef struct packed {
        logic                        is_cand; // status write: candidate vs terminal
        logic [NW_WIDTH-1:0]         wid;
        logic [RTU_SLOT_BITS-1:0]    slot;
        logic [NUM_LANES-1:0]        mask;    // write lane mask
        logic [NUM_LANES-1:0][31:0]  data;
        logic [TAG_WIDTH-1:0]        tag;     // routes the beat back to its core
    } win_data_t;

    logic       arm_valid;
    arm_data_t  arm_data;
    logic       arm_ready;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    logic       win_valid;
    win_data_t  win_data;
    logic       win_ready;

    // the window: sources arm/req, sinks win
    modport master (
        output arm_valid,
        output arm_data,
        input  arm_ready,

        output req_valid,
        output req_data,
        input  req_ready,

        input  win_valid,
        input  win_data,
        output win_ready
    );

    // the RTU core: sinks arm/req, sources win
    modport slave (
        input  arm_valid,
        input  arm_data,
        output arm_ready,

        input  req_valid,
        input  req_data,
        output req_ready,

        output win_valid,
        output win_data,
        input  win_ready
    );

endinterface
