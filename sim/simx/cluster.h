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

#include <simobject.h>
#include "cache.h"
#ifdef VX_CFG_EXT_DXA_ENABLE
#include "dxa_core.h"
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
#include "tex_core.h"
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om_core.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include "raster_core.h"
#endif

namespace vortex {

class ProcessorImpl;

class Socket;

class Cluster : public SimObject<Cluster> {
public:
  struct PerfStats {
    Cache::PerfStats l2cache;
#ifdef VX_CFG_EXT_DXA_ENABLE
    DxaCore::PerfStats dxa;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
    TexCore::PerfStats tex;
    Cache::PerfStats   tcache;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    RasterCore::PerfStats raster;
    Cache::PerfStats      rcache;
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    OmCore::PerfStats om;
    Cache::PerfStats  ocache;
#endif
  };

  std::vector<SimChannel<MemReq>> mem_req_out;
  std::vector<SimChannel<MemRsp>> mem_rsp_in;

  Cluster(const SimContext& ctx,
          const char* name,
          uint32_t cluster_id,
          ProcessorImpl* processor);

  ~Cluster();

  uint32_t id() const { return cluster_id_; }

  ProcessorImpl* processor() const { return processor_; }

  bool running() const;

  int get_exitcode() const;

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id);

  PerfStats perf_stats() const;

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  class Core* get_core(uint32_t idx) const;

  // Cache flush walk. The ProcessorImpl ticks the simulator in level
  // order (L1 surfaces in parallel → L2 → L3) to avoid downstream
  // evictions racing the next-level walk. Per-level the L1 fanout
  // mirrors RTL's VX_dcr_flush instances: dcache + icache +
  // {tcache, rcache, ocache} (the latter three gated by the matching
  // VX_CFG_EXT_* macros).
  void dcache_flush_begin();
  bool dcache_flush_done() const;
  void icache_flush_begin();
  bool icache_flush_done() const;
#ifdef VX_CFG_EXT_TEX_ENABLE
  void tcache_flush_begin();
  bool tcache_flush_done() const;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
  void rcache_flush_begin();
  bool rcache_flush_done() const;
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
  void ocache_flush_begin();
  bool ocache_flush_done() const;
#endif
  void l2_flush_begin();
  bool l2_flush_done() const;

#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core();
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Cluster-shared raster engine. Exposed so per-core SFU units can
  // dispatch the vx_rast_begin trigger (RasterType::BEGIN) directly —
  // the trigger is a side-band signal in the RTL (begin_pulse,
  // OR-reduced from every participating core's SFU), modelled here as
  // an on_begin() call into raster_core->begin().
  RasterCore::Ptr& raster_core();
#endif

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t       cluster_id_;
  ProcessorImpl* processor_;

  class Impl;
  Impl* impl_;

  friend class SimObject<Cluster>;
};

} // namespace vortex
