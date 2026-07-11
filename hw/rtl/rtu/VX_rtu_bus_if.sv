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

// VX_rtu_bus_if — per-core graphics window <-> socket-shared RTU core channel.
//
// The RTU is the MASTER of the window: the window is a passive slot register
// file that the RTU reads a ray out of and writes a hit record back into, one
// slot per beat, each beat carrying its own {wid, tbase, slot} address. The
// window therefore holds no ray, no hit record, and no traversal state — only
// the RAM and three per-warp status bits.
//
// Addressing the beats individually rather than bursting from a {base, length}
// descriptor costs ~12 address bits against a 32*NUM_LANES-bit data beat, which
// is nothing, and buys the window a stateless random-access port instead of a
// burst engine.
//
// Three channels, sized so that no channel can head-of-line block another:
//
//   arm  (window -> RTU)  A warp armed a TRACE. Carries the warp's identity and
//                         the warp-uniform scene base; the ray itself stays in
//                         the window until the RTU reads it. MAY BLOCK: the RTU
//                         accepts an arm only when idle, and nothing waits on it
//                         (the arming warp is wstall'd at decode).
//
//   req  (window -> RTU)  Everything the RTU is waiting FOR, so the RTU is
//                         ALWAYS ready here and this channel can never stall:
//                           CONT  — the warp's per-lane CONTINUE actions,
//                                   resuming an open candidate
//                           RDATA — a slot read return (the answer to a `win`
//                                   read)
//                         CONT and RDATA can share one channel because they are
//                         never outstanding at the same time: a CONT is what
//                         makes the RTU issue its next read.
//
//   win  (RTU -> window)  One slot access per beat: a read (answered on `req`
//                         as RDATA) or a masked write. Writes of one response
//                         are an ordered stream, so the status slot — written
//                         LAST — completes the parked WAIT only once the record
//                         is whole.
//
// Splitting `arm` out is what makes this deadlock-free. If an arm shared the
// `req` channel, a second warp's arm could sit at the arbiter head while the
// RTU refused it (busy) and the active warp's CONT queued behind it — the RTU
// waiting for a CONT that can never arrive.

interface VX_rtu_bus_if import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();
    // ── arm: a warp armed a TRACE ─────────────────────────────────────
    typedef struct packed {
        logic [NW_WIDTH-1:0]     wid;
        logic [RTU_TB_BITS-1:0]  tbase;      // thread base of the simd group
        logic [NUM_LANES-1:0]    mask;       // active lanes of the trace
        logic [31:0]             scene_base; // warp-uniform
        logic [TAG_WIDTH-1:0]    tag;
    } arm_data_t;

    // ── req: CONT actions | RDATA slot read return ────────────────────
    typedef struct packed {
        logic                                        kind;      // RTU_REQ_*
        logic [NUM_LANES-1:0][31:0]                  data;      // RDATA: slot word
        logic [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action; // CONT: per-lane action
    } req_data_t;

    // ── win: one window slot access ───────────────────────────────────
    typedef struct packed {
        logic                        we;      // 1 = masked write, 0 = read
        logic                        is_cand; // status write: candidate vs terminal
        logic [NW_WIDTH-1:0]         wid;
        logic [RTU_TB_BITS-1:0]      tbase;
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
