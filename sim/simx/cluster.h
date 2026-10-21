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
#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
#include "dtcu.h"
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
#ifdef VX_CFG_EXT_RTU_ENABLE
#include "rtu_core.h"
#endif

namespace vortex {

class ProcessorImpl;

class Socket;

class Cluster : public SimObject<Cluster> {
public:
  // Requesters at the cluster L2 fan-in, by name (RFC 261019c). Row numbers are a
  // construction detail of cluster.cpp and may be reordered.
  enum L2Requester : uint32_t {
    L2RQ_CORE = 0,   // socket L1 egress: dcache (incl. the socket DTCU's D stores and C preload) + icache
    L2RQ_DXA,
    L2RQ_DTCUC,      // cluster DTCU
    L2RQ_DTCUS,      // every socket DTCU, after the socket-engine funnel
    L2RQ_COUNT
  };

  struct L2RqStats {
    ArbPortStats arb;             // L2 fan-in arbiter outcome, summed over the request lanes
    uint64_t lost_to_core   = 0;  // arb.lost_sent whose winner was the core row
    uint64_t lost_to_engine = 0;  // arb.lost_sent whose winner was another non-core row
    uint64_t reads = 0, writes = 0, miss_r = 0, miss_w = 0;
    uint64_t xevicts = 0, evicted = 0;
  };

  struct PerfStats {
    Cache::PerfStats l2cache;
    std::array<L2RqStats, L2RQ_COUNT> l2rq{};
    uint64_t l2_pipe_cap = 0, l2_pipe_head = 0, l2_pipe_slot = 0;
    std::vector<ArbPortStats> l2xbar_lane;     // L2 bank crossbar, per L2 request lane
    std::vector<ArbPortStats> dtcu_sock_arb;   // socket-engine funnel, per socket index in the cluster
#ifdef VX_CFG_EXT_DXA_ENABLE
    DxaCore::PerfStats dxa;
#endif
#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
    Dtcu::PerfStats dtcu;
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
#ifdef VX_CFG_EXT_RTU_ENABLE
    RtuCore::PerfStats rtu;
    Cache::PerfStats   rtcache;
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

  // Cache flush walk. ProcessorImpl ticks in level order (L1 in parallel
  // → L2 → L3) to avoid downstream evictions racing the next-level walk.
  // L1 fanout: dcache + icache + {tcache, rcache, ocache} (gated by
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
#ifdef VX_CFG_EXT_RTU_ENABLE
  void rtcache_flush_begin();
  bool rtcache_flush_done() const;
#endif
  void l2_flush_begin();
  bool l2_flush_done() const;

#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core();
#endif

#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
  Dtcu::Ptr& dtcu();
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Cluster-shared raster engine (armed by the KMU's delegated draw launch;
  // the per-core SFU wave-pull launches covered-quad waves autonomously).
  RasterCore::Ptr& raster_core();
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
  RtuCore::Ptr& rtu_core();
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
