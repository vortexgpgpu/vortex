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
// See docs/proposals/rtu_simx_v3_proposal.md.
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

  // Phase 1: vx_rt_trace + vx_rt_wait are collapsed into one round-trip.
  // - process_trace builds the RtuReq from per-(warp,lane) regs and submits
  //   it (returns nullptr on backpressure). Also pre-clears dst_data; the
  //   handle is delivered as 0 (one in-flight per lane).
  // - process_wait is a no-op marker — the trace is owned by RtuCore until
  //   rsp arrival. SfuUnit's drain path applies the response into the RTU
  //   register file and forwards the trace to writeback with status word.
  instr_trace_t* process_trace(instr_trace_t* trace, uint32_t block_id);
  instr_trace_t* process_wait(instr_trace_t* trace, uint32_t block_id);

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

private:
  // RTU register file: per-(warp, lane, slot) 32-bit storage.
  static constexpr uint32_t SLOT_COUNT = VX_RT_SLOT_COUNT;
  using LaneRegs = std::array<uint32_t, SLOT_COUNT>;
  using WarpRegs = std::array<LaneRegs, VX_CFG_NUM_THREADS>;
  std::vector<WarpRegs> regfile_;  // [warp_id][lane][slot]

  Core*               core_;
  SimChannel<RtuReq>& req_out_;
};

} // namespace vortex
