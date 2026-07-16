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
//
// PRISM RTU (Ray-Tracing Unit).
//
// Architecture (mirrors TEX shape):
//   - RtuUnit is a per-core SFU PE driving the shared per-(warp,lane)
//     hit window (RtuWindow, owned by SfuUnit).
//   - The window ISA: vx_rt_wtrace streams the f0..f7 ray window into a
//     pool slot and sends the warp-packed RtuReq to RtuCore; vx_rt_wait is
//     the sync point that observes the matching RtuRsp. The callback-side
//     windowed reads/writes (GETWF/GETW/SETW) complete locally in 1 SFU cycle.

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <simobject.h>
#include <mempool.h>
#include "instr.h"
#include "instr_trace.h"
#include "constants.h"
#include "types.h"
#include "rtu_window.h"   // RtuWindow (the RTU hit-window slot file)
#include "rtu_types.h"   // RtuReq, RtuRsp, RtuReqKind, RtuRspKind,
                         // RtuBusArbiter (namespace vortex::rtu, with
                         // vortex:: aliases).

namespace vortex {

class Core;
class RtuCore;

///////////////////////////////////////////////////////////////////////////////

// Trace micro-op generator. Owned by each
// per-warp Sequencer; expands the TRACE / WAIT macro-ops into the uops that
// stream the f0..f7 ray window into the pool slot and retire the hit window.
// Mirrors TcuUopGen — the architectural encoding names only rd/rs1; the
// register windows ride HW convention materialized here.
class RtuUopGen {
public:
  RtuUopGen(PoolAllocator<Instr, 64>& pool) : pool_(pool) {}

  // Total micro-op count for a macro instruction (>1 means macro-op).
  //   TRACE -> 4  (1 GP config + 3 FP ray)
  //   WAIT  -> 7  (1 GP status + 3 FP hit + 3 GP id)
  static uint32_t uop_count(const Instr& instr);

  // Generate micro-op Instr at uop_index for the given macro instruction.
  Instr::Ptr get(const Instr& macro_instr, uint32_t uop_index);

private:
  PoolAllocator<Instr, 64>& pool_;
};

// Per-core SFU PE for the window ISA (vx_rt_wtrace / vx_rt_wait /
// vx_rt_get[w]f / vx_rt_set1 / vx_rt_cb_ret). Owns the per-(warp,lane) RTU
// register file. Plain (non-SimObject) helper owned by SfuUnit.
class RtuUnit {
public:
  RtuUnit(Core* core, SimChannel<RtuReq>& req_out, RtuWindow& window);

  // Async ray pool. process_wait either:
  //   - returns the trace with the per-lane status word written
  //     into dst_data — fast path, used when TERMINAL already
  //     landed (pending_terminals_) before WAIT issued. Caller does
  //     output.send(). The slot is freed here.
  //   - returns nullptr — slot has not yet completed. The trace is
  //     parked in wait_parked_; the matching TERMINAL drain in
  //     SfuUnit will pick it up via take_pending_writeback() once
  //     the cluster's RtuCore emits the terminal rsp.
  // Caller MUST pre-check wait_would_short_circuit() and reserve
  // an output slot before calling; otherwise the synchronous path
  // has no place to deliver.
  instr_trace_t* process_wait(instr_trace_t* trace, uint32_t block_id);

  // Handle that a WAIT trace will block on. Reads rs1 of the
  // first active lane (assumes warp-uniform
  // handles; the divergent case is a follow-up).
  static uint32_t wait_handle(const instr_trace_t* trace);

  // Would process_wait take the fast (short-circuit) path?
  // Used by SfuUnit to gate output.full() before calling
  // process_wait. Returns false (=> park-bound) when the slot's
  // TERMINAL hasn't landed yet.
  bool wait_would_short_circuit(uint32_t wid, uint32_t slot) const;

  // Called by SfuUnit when an RtuRsp lands. If a matching
  // wait_parked_ entry exists, returns the parked trace + its
  // block_id and frees the slot; the caller then output.sends the
  // trace. If no wait is parked yet, latches the rsp into
  // pending_terminals_ and returns nullptr.
  struct PendingWriteback {
    instr_trace_t* trace;
    uint32_t       block_id;
  };
  PendingWriteback on_terminal_rsp(const RtuRsp& rsp);

  // Peek whether on_terminal_rsp(rsp) would return a
  // writeback (true) or latch the rsp silently (false). If true,
  // also fills *out_block_id with the parked WAIT's output block
  // so SfuUnit can pre-check output.full() before calling
  // on_terminal_rsp (which is destructive — it frees the slot and
  // erases the parked entry).
  bool terminal_would_writeback(const RtuRsp& rsp, uint32_t* out_block_id) const;

  // Candidate-return counterparts of on_terminal_rsp / terminal_would_writeback.
  // A non-opaque candidate (AHS / procedural) completes the parked WAIT with a
  // YIELD status but leaves the slot live (traversal resumes on CONTINUE).
  PendingWriteback on_candidate_rsp(const RtuRsp& rsp);
  bool candidate_would_writeback(const RtuRsp& rsp, uint32_t* out_block_id) const;

  // vx_rt_continue emits the per-lane action for the returned candidate. Reads
  // the action code from rs1 and emits a CB_ACTION packet through the bus to
  // RtuCore. Returns nullptr on backpressure (caller retries next cycle); else
  // the trace, which the SFU forwards to writeback.
  instr_trace_t* process_cb_ret(instr_trace_t* trace, uint32_t block_id);

  // One micro-op of a TRACE macro:
  //   uop 0 — read lane-packed config (rs1), allocate a pool slot, write the
  //           handle to dst, stage flags/cull/payload/scene.
  //   uop 1..2 — stream origin / direction from the f0..f5 window into the
  //           staged ray slots.
  //   uop 3 — stream tmin/tmax (f6/f7), then ARM the slot (build + send the
  //           RtuReq). Returns nullptr on backpressure (pool full at uop 0,
  //           bus full at uop 3); else the trace.
  instr_trace_t* process_trace_uop(instr_trace_t* trace, uint32_t block_id, uint32_t uop);

  // One micro-op of a WAIT macro:
  //   uop 0 — identical to WAIT (park until terminal / short-circuit); the
  //           terminal rsp stages the hit attrs into regfile_ via
  //           apply_response. Returns nullptr when parked (same contract as
  //           process_wait).
  //   uop 1..6 — issue only after uop 0 retires (scoreboard-chained on the
  //           status reg); copy one staged hit attr from regfile_ into the
  //           uop's dst register (t/u/v -> FP, IDs -> GP). Always return the
  //           trace.

  // Apply a TERMINAL RtuRsp into the RTU register file (hit_t, hit
  // attrs, IDs). Called by SfuUnit at rsp drain.
  void apply_response(const RtuRsp& rsp);

  // Apply a candidate (CB_YIELD) RtuRsp's candidate-hit attrs into the RTU
  // register file for the yielded lanes, so the warp's any-hit / intersection
  // code reads the right payload before issuing CONTINUE.
  void apply_callback_payload(const RtuRsp& rsp);

  // Representative slot of a candidate rsp (first active lane's cb_handle).
  static uint32_t candidate_slot(const RtuRsp& rsp);

  // Per-lane status writeback for a completing WAIT/CONTINUE (see .cpp): a
  // candidate leaves its non-yielding lanes PENDING (still traversing).
  static void write_status(instr_trace_t* trace, const RtuRsp& rsp, bool is_candidate);

  // Async ray pool: Cluster wires this after RtuCore exists so
  // RtuUnit can directly call allocate_slot()/free_slot() on the
  // shared cluster-level pool (no SimChannel hop). Both pointers are
  // borrowed — RtuCore outlives RtuUnit (Cluster owns both).
  void set_rtu_core(RtuCore* core) { rtu_core_ = core; }

  // Claim this warp's slot in the cluster-shared ray pool, at issue. A TRACE2
  // whose head uop entered the SFU without a slot would stall at the head of
  // the unit's queue, behind which sits the WAIT2 that is the only way a slot
  // is ever released. Returns false when the pool is full, and the issue stage
  // holds the warp.
  bool trace2_reserve_slot(uint32_t wid);

private:
  // The hit window (see rtu/rtu_window.h): traversal RESULTS only. The RTU is its
  // only writer, the shader its only reader, and the RTU never reads it back. The
  // response paths address it as window_.warp(wid)[lane][slot].
  RtuWindow&          window_;

  Core*               core_;
  SimChannel<RtuReq>& req_out_;
  // Async ray pool. Borrowed from the Socket via set_rtu_core();
  // null until the Socket has wired it (TRACE/WAIT paths must NEVER
  // dereference rtu_core_ before that — but in practice the Socket
  // calls set_rtu_core() at construction time, before any TRACE
  // can dispatch). Single shared pool per socket — alloc/free is
  // contended across that socket's per-core RtuUnits.
  RtuCore*            rtu_core_ = nullptr;

  // WAIT-park bookkeeping. Both tables are keyed by slot
  // handle and indexed by warp_id. wait_parked_ holds WAIT traces
  // whose TERMINAL hasn't landed yet; pending_terminals_ holds
  // TERMINAL rsps that landed before their WAIT issued (rare but
  // possible — short rays + late-arriving WAIT). Exactly one of
  // the two has an entry for any (wid, slot) at any time.
  struct ParkedWait { instr_trace_t* trace; uint32_t block_id; };
  std::array<std::unordered_map<uint32_t, ParkedWait>,
             VX_CFG_NUM_WARPS>           wait_parked_;
  std::array<std::unordered_map<uint32_t, RtuRsp>,
             VX_CFG_NUM_WARPS>           pending_terminals_;

  // Lane mask of the candidate most recently returned to each warp. The warp's
  // CONTINUE applies its actions to exactly these lanes: the SIMT mask at the
  // CONTINUE also carries lanes that are merely PENDING (still traversing), and
  // those may have a candidate queued for a later batch, so their garbage action
  // must not be allowed to resolve it.
  std::array<uint32_t, VX_CFG_NUM_WARPS>  last_cand_mask_{};

  // Per-warp cross-uop TRACE state. The ray NEVER enters the window: the burst
  // stages it here and hands it to RtuCore at the arm. The window is result
  // storage — the RTU writes it, the shader reads it, nothing else touches it.
  struct TraceRay {
    std::array<uint32_t, 3> origin{};
    std::array<uint32_t, 3> dir{};
    uint32_t t_min = 0;
    uint32_t t_max = 0;
  };
  std::array<int32_t, VX_CFG_NUM_WARPS>  trace_slot_;
  std::array<std::array<TraceRay, VX_CFG_NUM_THREADS>,
             VX_CFG_NUM_WARPS>           trace_ray_;
  // Warp-uniform half of the ray, staged by the config uop: it rides the arm
  // doorbell in HW, so it is not a window write either.
  std::array<std::array<uint32_t, VX_CFG_NUM_THREADS>,
             VX_CFG_NUM_WARPS>           trace_scene_;
  std::array<uint32_t, VX_CFG_NUM_WARPS> trace_payload_{};
  std::array<uint32_t, VX_CFG_NUM_WARPS> trace_flags_{};
  std::array<uint32_t, VX_CFG_NUM_WARPS> trace_cull_{};
  // The slot handle each lane's outstanding candidate came from, mirrored here
  // when the candidate is delivered. The CONTINUE routes its action back by this
  // handle (same-warp reformation can bundle lanes from several slots into one
  // candidate), and it is read from here, not from the window — the RTU never
  // reads the window.
  std::array<std::array<uint32_t, VX_CFG_NUM_THREADS>,
             VX_CFG_NUM_WARPS>           cb_handle_{};
};

} // namespace vortex
