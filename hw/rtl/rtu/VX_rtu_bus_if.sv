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

// VX_rtu_bus_if — per-core SFU shim ↔ cluster-shared RTU core channel.
//
// Both directions are BEAT-SERIAL: one `[NUM_LANES][31:0]` word per cycle, with
// `eop` marking the last beat of a transfer. A ray is 10 beats, a hit record 8
// (terminal) or 10 (yield). Everything else is small sideband held stable for
// the whole transfer.
//
// Serial rather than parallel because the payload lives in a memory at one end
// (the graphics window's slot RAM) and in per-lane traversal contexts at the
// other. A parallel payload would force both endpoints, plus every buffer and
// arbiter between them, to materialize a whole ray in flip-flops — several
// thousand per lane, on a channel that carries one ray every few hundred
// cycles. Streaming reads it straight out of one memory and into the other.
//
// A transfer must not be interleaved with another requester's: the arbiters
// hold their grant for the duration (VX_stream_arb STICKY).
//
// Shader callbacks overload both directions with a kind tag:
//   req.kind = TRACE  — 10 ray beats (origin, dir, t_min, t_max, flags, cull);
//                       `scene_base` rides sideband (it is warp-uniform).
//            = CBACT  — a single beat carrying the IS-computed t; the per-lane
//                       action rides `cb_action` sideband.
//   rsp.kind = TERMINAL — 7 hit-attribute beats, then the status word.
//            = CB_YIELD — the same 7 candidate attributes, then cb_type,
//                       cb_sbt_idx and the callback handle. `cb_active_mask`
//                       marks the yielding lanes and holds for the transfer.
// Beat order is fixed by RTU_REQ_BEAT_* / RTU_RSP_BEAT_* in VX_rtu_pkg; both
// endpoints index the same tables.

interface VX_rtu_bus_if import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();
    typedef struct packed {
        logic                                        kind;     // RTU_REQ_*
        logic                                        eop;      // last beat
        logic [NUM_LANES-1:0]                         mask;
        logic [NUM_LANES-1:0][31:0]                   data;     // beat word
        logic [NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cb_action;// CBACT sideband
        logic [`VX_CFG_MEM_ADDR_WIDTH-1:0]            scene_base; // TRACE sideband
        logic [TAG_WIDTH-1:0]                         tag;
    } req_data_t;

    typedef struct packed {
        logic                            kind;   // RTU_RSP_*
        logic                            eop;    // last beat
        logic [NUM_LANES-1:0][31:0]      data;   // beat word
        // CB_YIELD only — yielding-lane mask, held for the whole transfer.
        logic [NUM_LANES-1:0]            cb_active_mask;
        logic [TAG_WIDTH-1:0]            tag;
    } rsp_data_t;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    logic       rsp_valid;
    rsp_data_t  rsp_data;
    logic       rsp_ready;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready,

        input  rsp_valid,
        input  rsp_data,
        output rsp_ready
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready,

        output rsp_valid,
        output rsp_data,
        input  rsp_ready
    );

endinterface
