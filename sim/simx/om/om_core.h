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

#pragma once

#include <memory>
#include <simobject.h>
#include "types.h"
#include "om_unit.h"

namespace vortex {

class Cluster;

// Cluster-shared OM (output-merger) engine. Mirrors RTL VX_om_core:
//   om_arb (cluster) → VX_om_core { dcr, ds, blend, mem }
// Receives per-core OmReqs, drives R-M-W against the ocache through
// MemReq/MemRsp, and applies the depth/stencil + blend pipelines from
// graphics::DepthTencil + graphics::Blender on response arrival
// (functional + timing in one place per simx_v3 §3.3).
class OmCore : public SimObject<OmCore> {
public:
  using Ptr = std::shared_ptr<OmCore>;

  struct PerfStats {
    uint64_t mem_reads    = 0;
    uint64_t mem_writes   = 0;
    uint64_t mem_latency  = 0;
    uint64_t stall_cycles = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      mem_reads    += rhs.mem_reads;
      mem_writes   += rhs.mem_writes;
      mem_latency  += rhs.mem_latency;
      stall_cycles += rhs.stall_cycles;
      return *this;
    }
  };

  // Per-core inbound OmReq channels (size = NUM_CORES_PER_CLUSTER).
  // Cluster binds each core's SfuUnit::om_req_out here.
  std::vector<SimChannel<OmReq>>  om_req_in;

  // Memory ports to the ocache. Size = OCACHE_NUM_REQS.
  std::vector<SimChannel<MemReq>> ocache_req_out;
  std::vector<SimChannel<MemRsp>> ocache_rsp_in;

  OmCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~OmCore();

  int dcr_write(uint32_t addr, uint32_t value);

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<OmCore>;
};

} // namespace vortex
