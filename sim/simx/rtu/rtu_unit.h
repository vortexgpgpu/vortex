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
// PRISM RTU (Ray-Tracing Unit) — Phase 1.
// See docs/proposals/rtu_simx_proposal.md.
//
// Architecture (mirrors TEX shape exactly for Phase 1):
//   - RtuUnit is a per-core SFU PE owning the per-(warp,lane) RTU register
//     file (29 named 32-bit slots × NUM_WARPS × NUM_THREADS).
//   - vx_rt_set / vx_rt_get complete locally in 1 SFU cycle.
//   - vx_rt_trace + vx_rt_wait both flow through RtuCore via warp-packed
//     RtuReq/RtuRsp packets — same pattern as TEX. Phase 1 collapses trace
//     and wait into one round-trip since Phase 1 has no per-lane handle
//     map yet (one outstanding ray per (warp,lane)). The trace op snapshots
//     all active lanes' ray descriptors into the RtuReq and sends it; the
//     wait op acts as the sync point that observes the matching RtuRsp.

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <simobject.h>
#include "instr_trace.h"
#include "constants.h"
#include "types.h"
#include "rtu_types.h"   // §step-2 refactor: RtuReq, RtuRsp, RtuReqKind,
                         // RtuRspKind, RtuBusArbiter now live in rtu_types.h
                         // under namespace vortex::rtu, with vortex:: aliases.

namespace vortex {

class Core;
class RtuCore;

// Per-core SFU PE for vx_rt_set / vx_rt_get / vx_rt_trace / vx_rt_wait /
// vx_rt_cb_ret. Owns the per-(warp,lane) RTU register file. Plain
// (non-SimObject) helper owned by SfuUnit.
class RtuUnit {
public:
  RtuUnit(Core* core, SimChannel<RtuReq>& req_out);

  // Synchronous handlers — complete in 1 SFU cycle.
  instr_trace_t* process_set(instr_trace_t* trace);
  instr_trace_t* process_get(instr_trace_t* trace);

  // §8.6 async ray pool: vx_rt_trace pre-allocates a slot in
  // RtuCore's pool and writes the slot index back as the handle —
  // synchronously, via the standard SFU writeback. The ray work
  // runs async in RtuCore; vx_rt_wait does the actual blocking.
  // Returns nullptr on backpressure (pool full OR bus full); else
  // the trace, which the SFU forwards to writeback (so the handle
  // becomes visible to the kernel).
  instr_trace_t* process_trace(instr_trace_t* trace, uint32_t block_id);

  // §8.6 async ray pool. process_wait either:
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

  // §8.6: handle that a WAIT trace will block on. Reads rs1 of the
  // first active lane (Phase-1 of §8.6 assumes warp-uniform
  // handles; the divergent case is a follow-up).
  static uint32_t wait_handle(const instr_trace_t* trace);

  // §8.6: would process_wait take the fast (short-circuit) path?
  // Used by SfuUnit to gate output.full() before calling
  // process_wait. Returns false (=> park-bound) when the slot's
  // TERMINAL hasn't landed yet.
  bool wait_would_short_circuit(uint32_t wid, uint32_t slot) const;

  // §8.6: called by SfuUnit when an RtuRsp lands. If a matching
  // wait_parked_ entry exists, returns the parked trace + its
  // block_id and frees the slot; the caller then output.sends the
  // trace. If no wait is parked yet, latches the rsp into
  // pending_terminals_ and returns nullptr.
  struct PendingWriteback {
    instr_trace_t* trace;
    uint32_t       block_id;
  };
  PendingWriteback on_terminal_rsp(const RtuRsp& rsp);

  // §8.6: peek whether on_terminal_rsp(rsp) would return a
  // writeback (true) or latch the rsp silently (false). If true,
  // also fills *out_block_id with the parked WAIT's output block
  // so SfuUnit can pre-check output.full() before calling
  // on_terminal_rsp (which is destructive — it frees the slot and
  // erases the parked entry).
  bool terminal_would_writeback(const RtuRsp& rsp, uint32_t* out_block_id) const;

  // Phase 2: vx_rt_cb_ret releases this warp's parked callback. Reads
  // per-lane action code from rs1 and emits a CB_ACTION packet through
  // the bus to RtuCore. Returns nullptr on backpressure (caller retries
  // next cycle); else the trace, which the SFU forwards to writeback.
  instr_trace_t* process_cb_ret(instr_trace_t* trace, uint32_t block_id);

  // Apply a TERMINAL RtuRsp into the RTU register file (hit_t, hit
  // attrs, IDs). Called by SfuUnit at rsp drain.
  void apply_response(const RtuRsp& rsp);

  // Apply a CB_YIELD RtuRsp's candidate-hit attrs into the RTU register
  // file for the yielded lanes. Called by SfuUnit before raising the
  // async trap into the callback dispatcher.
  void apply_callback_payload(const RtuRsp& rsp);

  // §8.6 async ray pool: Cluster wires this after RtuCore exists so
  // RtuUnit can directly call allocate_slot()/free_slot() on the
  // shared cluster-level pool (no SimChannel hop). Both pointers are
  // borrowed — RtuCore outlives RtuUnit (Cluster owns both).
  void set_rtu_core(RtuCore* core) { rtu_core_ = core; }

private:
  // RTU register file: per-(warp, lane, slot) 32-bit storage.
  static constexpr uint32_t SLOT_COUNT = VX_RT_SLOT_COUNT;
  using LaneRegs = std::array<uint32_t, SLOT_COUNT>;
  using WarpRegs = std::array<LaneRegs, VX_CFG_NUM_THREADS>;
  std::vector<WarpRegs> regfile_;  // [warp_id][lane][slot]

  Core*               core_;
  SimChannel<RtuReq>& req_out_;
  // §8.6 async ray pool. Borrowed from Cluster via set_rtu_core();
  // null until Cluster has wired it (TRACE/WAIT paths must NEVER
  // dereference rtu_core_ before that — but in practice Cluster
  // calls set_rtu_core() at construction time, before any TRACE
  // can dispatch). Single shared pool per cluster — alloc/free is
  // contended across all per-core RtuUnits.
  RtuCore*            rtu_core_ = nullptr;

  // §8.6 WAIT-park bookkeeping. Both tables are keyed by slot
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
};

} // namespace vortex
