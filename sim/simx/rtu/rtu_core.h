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

class Cluster;

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

  RtuCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~RtuCore();

  const PerfStats& perf_stats() const;

  // §8.6 async ray pool. The per-core RtuUnit calls allocate_slot()
  // at TRACE-issue time so vx_rt_trace can writeback a real handle
  // (= slot index) instead of the Phase-1 fixed 0. The same RtuUnit
  // calls free_slot() at WAIT-completion time (after TERMINAL has
  // been delivered to the kernel) so the slot returns to the pool.
  // Both calls are direct C++ — no SimChannel hop — because there's
  // no per-tick ordering concern: the allocator just tracks
  // in_use bits in the SlotPool.
  //
  // Returns the slot index on success, or -1 if every slot in the
  // pool is currently allocated (caller must retry next cycle).
  int32_t allocate_slot();
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
