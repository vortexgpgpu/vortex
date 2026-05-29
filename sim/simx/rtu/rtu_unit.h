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

namespace vortex {

class Core;

// Two kinds of packet share the RtuReq channel:
//   TRACE_NEW   — vx_rt_trace fires a fresh ray (Phase 1 path).
//   CB_ACTION   — vx_rt_cb_ret releases a parked context with an action
//                 (ACCEPT/IGNORE/TERMINATE) per lane. Phase 2.
enum class RtuReqKind : uint8_t {
  TRACE_NEW = 0,
  CB_ACTION = 1,
};

// RtuReq — per-warp packet. Carries the snapshot of ray-input slots for
// every active lane (TRACE_NEW path) or the per-lane cb_ret action codes
// (CB_ACTION path). Simulator-only fields ride alongside for write-back
// routing.
struct RtuReq {
  RtuReqKind kind = RtuReqKind::TRACE_NEW;
  uint64_t uuid = 0;
  uint32_t tag  = 0;
  uint32_t tmask_bits = 0;

  // Per-lane ray descriptor snapshot (TRACE_NEW only).
  std::array<uint32_t, VX_CFG_NUM_THREADS> scene_root = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_x   = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_y   = {};
  std::array<float,    VX_CFG_NUM_THREADS> origin_z   = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_x      = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_y      = {};
  std::array<float,    VX_CFG_NUM_THREADS> dir_z      = {};
  std::array<float,    VX_CFG_NUM_THREADS> tmin       = {};
  std::array<float,    VX_CFG_NUM_THREADS> tmax       = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> flags      = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cull_mask  = {};

  // Per-lane cb_ret action codes (CB_ACTION only). One of VX_RT_CB_*.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_action  = {};

  // Per-lane RtuCore slot handle (CB_ACTION only) — read from the kernel's
  // VX_RT_CB_HANDLE slot at vx_rt_cb_ret time. Phase 3-A2 reformation may
  // batch lanes from MULTIPLE slots into one virtual warp at CB_YIELD, so
  // the action packet has to route per-lane back to the originating slot
  // rather than rely on a single warp-scoped slot id.
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_handle  = {};

  // SimX-only: routing back to per-core SfuUnit writeback.
  instr_trace_t* trace    = nullptr;
  uint32_t       block_id = 0;
  uint32_t       warp_id  = 0;

  RtuReq() = default;

  friend std::ostream& operator<<(std::ostream& os, const RtuReq& req) {
    os << (req.kind == RtuReqKind::TRACE_NEW ? "TRACE" : "CB_RET")
       << " tag=0x" << std::hex << req.tag << std::dec
       << ", tmask=0x" << std::hex << req.tmask_bits << std::dec
       << " (#" << req.uuid << ")";
    return os;
  }
};

// Two kinds of response share the RtuRsp channel:
//   TERMINAL    — the request has finished (HIT/MISS). Per-lane status
//                 and hit attributes are populated; the trace is forwarded
//                 to SFU writeback. Phase 1 / Phase 2 happy path.
//   CB_YIELD    — Phase 2: one or more lanes hit a non-opaque triangle
//                 (AHS) or procedural AABB (IS). cb_active_mask marks
//                 which lanes need the callback; cb_type / candidate-hit
//                 attrs are populated for those lanes. SfuUnit's drain
//                 path stages those attrs into the RTU regs and raises
//                 an async trap on the warp; the trace stays parked at
//                 the vx_rt_wait and is not forwarded to writeback until
//                 a later TERMINAL rsp arrives.
enum class RtuRspKind : uint8_t {
  TERMINAL = 0,
  CB_YIELD = 1,
};

struct RtuRsp {
  RtuRspKind kind = RtuRspKind::TERMINAL;
  uint64_t uuid = 0;
  uint32_t tag  = 0;

  // Per-lane terminal status + hit attributes (TERMINAL: VX_RT_STS_DONE_*;
  // CB_YIELD: candidate-hit attrs for yielded lanes only).
  std::array<uint32_t, VX_CFG_NUM_THREADS> status            = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_t             = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_u        = {};
  std::array<float,    VX_CFG_NUM_THREADS> hit_bary_v        = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_primitive_id  = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_instance_id   = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> hit_geometry_index = {};

  // CB_YIELD only: which lanes are yielding (bitmask) and what kind of
  // callback each yielding lane needs (VX_RT_CB_TYPE_*). cb_handle is the
  // per-lane slot id (Phase 3-A2) so the kernel can write it into the
  // VX_RT_CB_HANDLE slot and route the matching CB_RET back per-lane.
  uint32_t cb_active_mask = 0;
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_type    = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_handle  = {};
  std::array<uint32_t, VX_CFG_NUM_THREADS> cb_sbt_idx = {};

  instr_trace_t* trace    = nullptr;
  uint32_t       block_id = 0;
  uint32_t       warp_id  = 0;

  RtuRsp() = default;
  RtuRsp(const RtuReq& req)
    : uuid(req.uuid), tag(req.tag),
      trace(req.trace), block_id(req.block_id), warp_id(req.warp_id) {}

  friend std::ostream& operator<<(std::ostream& os, const RtuRsp& rsp) {
    os << (rsp.kind == RtuRspKind::TERMINAL ? "DONE" : "CB_YIELD")
       << " tag=0x" << std::hex << rsp.tag << std::dec
       << " (#" << rsp.uuid << ")";
    return os;
  }
};

using RtuBusArbiter = TxRxArbiter<RtuReq, RtuRsp>;

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
