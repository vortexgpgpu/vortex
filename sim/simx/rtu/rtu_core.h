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
// PRISM RtuCore — Phase 1 minimum.
//
// Cluster-scope SimObject that consumes RtuReq packets from per-core RtuUnits
// and produces RtuRsp packets on completion. Phase 1 implements a flat
// "scene" walk: the TLAS device address points to a simple_scene_t with a
// uint32 triangle_count followed by N triangles (9 floats each). RtuCore
// issues dcache loads for the scene, runs ray-triangle intersection across
// the triangle list, picks the closest opaque hit, and emits an RtuRsp with
// VX_RT_STS_DONE_HIT (with hit_t / barycentrics / primitive_id) or
// VX_RT_STS_DONE_MISS.
//
// Phase 2 will replace the flat-scene walker with a real CW-BVH4 traversal
// and add shader queues; Phase 3 adds reformation.

#pragma once

#include <memory>
#include <simobject.h>
#include "types.h"
#include "rtu_types.h"  // §step-2: PerfStats now in vortex::rtu namespace
#include "rtu_unit.h"

namespace vortex {

class Socket;

class RtuCore : public SimObject<RtuCore> {
public:
  using Ptr = std::shared_ptr<RtuCore>;

  // §step-2 refactor: PerfStats moved to rtu_types.h
  // (vortex::rtu::PerfStats). RtuCore::PerfStats remains a stable
  // back-compat alias so Cluster::PerfStats::rtu can stay typed as
  // RtuCore::PerfStats and external callers don't break.
  using PerfStats = ::vortex::rtu::PerfStats;

  // Inputs from per-socket RtuBus arbiter (cluster collapses sockets → 1).
  std::vector<SimChannel<RtuReq>>  rtu_req_in;
  std::vector<SimChannel<RtuRsp>>  rtu_rsp_out;

  // Memory ports to the cluster dcache cluster. Size = NUM_RTU_BLOCKS.
  std::vector<SimChannel<MemReq>>  dcache_req_out;
  std::vector<SimChannel<MemRsp>>  dcache_rsp_in;

  RtuCore(const SimContext& ctx, const char* name, Socket* socket);
  virtual ~RtuCore();

  const PerfStats& perf_stats() const;

  // The slot pool's credit gate. The per-core RtuUnit claims a slot at
  // TRACE-issue time — before the macro-op enters the SFU — so a full pool
  // stalls the warp at issue instead of jamming the in-order SFU head behind a
  // TRACE that only a WAIT queued behind it could ever release. The claim also
  // gives vx_rt_wtrace a real handle to write back. free_slot() returns the slot
  // once the WAIT has consumed the record.
  //
  // Slots partition per core (RTU_SLOTS_PER_CORE), so one core cannot spend
  // another's share. Returns the slot index, or -1 if this core is already at
  // its quota — the caller retries next cycle.
  int32_t allocate_slot(uint32_t core_id);
  void    free_slot(uint32_t slot_idx);

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<RtuCore>;
};

} // namespace vortex
