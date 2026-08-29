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
#include "types.h"
#include "raster_unit.h"

namespace vortex {

class Cluster;

// Cluster-shared RASTER engine. On the first wave request, walks the host-built
// tile/primitive buffers via the rcache (MemReq/MemRsp), runs
// graphics::Rasterizer to enumerate every covered quad's pos_mask, then
// serves per-core pop requests from the internal queue. Returns done=0
// (encoded as 0) when drained.
class RasterCore : public SimObject<RasterCore> {
public:
  using Ptr = std::shared_ptr<RasterCore>;

  struct PerfStats {
    uint64_t mem_reads    = 0;
    uint64_t mem_latency  = 0;
    uint64_t stall_cycles = 0;
    uint64_t raster_cycles = 0;   // TE/BE walker pipeline cycles (quad-emission rate)
    uint64_t earlyz_tested = 0;   // P3: covered pixels tested by early-Z
    uint64_t earlyz_culled = 0;   // P3: covered pixels culled before shading

    PerfStats& operator+=(const PerfStats& rhs) {
      mem_reads    += rhs.mem_reads;
      mem_latency  += rhs.mem_latency;
      stall_cycles += rhs.stall_cycles;
      raster_cycles += rhs.raster_cycles;
      earlyz_tested += rhs.earlyz_tested;
      earlyz_culled += rhs.earlyz_culled;
      return *this;
    }
  };

  // Cluster-level inbound request / outbound response (one lane per cluster).
  // RasterBusArbiter fans per-core inputs into this lane and routes
  // responses back per-core.
  std::vector<SimChannel<RasterReq>>  raster_req_in;
  std::vector<SimChannel<RasterRsp>>  raster_rsp_out;

  // Memory ports to the rcache. Size = RCACHE_NUM_REQS.
  std::vector<SimChannel<MemReq>>     rcache_req_out;
  std::vector<SimChannel<MemRsp>>     rcache_rsp_in;

  // Fragment-work-distributor control links: one arm strobe per owned core;
  // every core reports its drain back on the shared done link.
  std::vector<SimEventLink<FwdArm>>   fwd_arm_out;
  SimEventLink<FwdDone>               fwd_done_in;

  RasterCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~RasterCore();

  // Writing the RASTER config re-arms the producer for the new frame (no
  // separate begin op); the FSM kicks off the tile/prim load on the first
  // pulled wave.
  int dcr_write(uint32_t addr, uint32_t value);

  // Snoop OM depth DCRs (shared depth-buffer config) for the early-Z stage.
  // Called by Cluster::dcr_write on OM-range writes; unlike dcr_write it does
  // not re-arm the producer.
  void om_dcr_snoop(uint32_t addr, uint32_t value);

  // Fragment-shader dispatch descriptor (RASTER_FRAG_* DCRs). Stored here
  // because it must persist across the per-launch SimPlatform reset (like the
  // KMU descriptor); consumed when the delegated launch arms the frame.
  void set_frag_descriptor(uint64_t frag_entry, uint64_t frag_param);

  // Frame kick — the KMU's delegated draw launch (grid-less kernel launch).
  // Arrives after the draw's DCR sequence and after the per-launch reset;
  // arms the owned cores' fragment work distributors on the next tick.
  void frame_kick();

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  void on_fwd_done(const FwdDone& msg);

  class Impl;
  Impl* impl_;

  friend class SimObject<RasterCore>;
};

} // namespace vortex
