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
#include "raster_unit.h"

namespace vortex {

class Cluster;

// Cluster-shared RASTER engine. Mirrors RTL VX_raster_core:
//   raster_arb (cluster) → VX_raster_core { dcr, mem, slice }
// On first vx_rast(), walks the host-built tile/primitive buffers via the
// rcache (MemReq/MemRsp), runs graphics::Rasterizer to enumerate every
// covered quad's pos_mask, then serves per-core pop requests from the
// internal queue. Returns done=0 (encoded as 0) when drained.
class RasterCore : public SimObject<RasterCore> {
public:
  using Ptr = std::shared_ptr<RasterCore>;

  struct PerfStats {
    uint64_t mem_reads    = 0;
    uint64_t mem_latency  = 0;
    uint64_t stall_cycles = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      mem_reads    += rhs.mem_reads;
      mem_latency  += rhs.mem_latency;
      stall_cycles += rhs.stall_cycles;
      return *this;
    }
  };

  // Cluster-level inbound request / outbound response. Size = 1 (one
  // RasterCore per cluster after the cluster-level arb collapses NUM_SLICES
  // → 1). The RasterBusArbiter fans NUM_CORES_PER_CLUSTER per-core inputs
  // into this single lane and routes responses back per-core.
  std::vector<SimChannel<RasterReq>>  raster_req_in;
  std::vector<SimChannel<RasterRsp>>  raster_rsp_out;

  // Memory ports to the rcache. Size = RCACHE_NUM_REQS.
  std::vector<SimChannel<MemReq>>     rcache_req_out;
  std::vector<SimChannel<MemRsp>>     rcache_rsp_in;

  RasterCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~RasterCore();

  int dcr_write(uint32_t addr, uint32_t value);

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<RasterCore>;
};

} // namespace vortex
