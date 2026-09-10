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

#include "types.h"
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
#ifdef VX_CFG_EXT_RTU_ENABLE
#include "rtu_core.h"
#endif
#ifdef VX_CFG_VM_ENABLE
#include "mem/tlb_l2.h"
#include "mem/ptw.h"
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
#ifdef VX_CFG_EXT_RTU_ENABLE
    RtuCore::PerfStats rtu;
    Cache::PerfStats   rtcache;
#endif
#ifdef VX_CFG_VM_ENABLE
    L2Tlb::PerfStats l2tlb;
    Ptw::PerfStats   ptw;
#endif
  };

  std::vector<SimChannel<MemReq>> mem_req_out;
  std::vector<SimChannel<MemRsp>> mem_rsp_in;

  // Global-barrier event links: every core's arrive end fans into
  // gbar_arrive_in; resume links fan back out, one per cluster core.
  SimEventLink<GbarArrive> gbar_arrive_in;
  std::vector<SimEventLink<GbarResume>> gbar_resume_out;

  Cluster(const SimContext& ctx,
          const char* name,
          uint32_t cluster_id,
          ProcessorImpl* processor);

  ~Cluster();

  uint32_t id() const { return cluster_id_; }

  ProcessorImpl* processor() const { return processor_; }

  bool running() const;

  int get_exitcode() const;

  PerfStats perf_stats() const;

  int dcr_write(uint32_t addr, uint32_t value);

#ifdef VX_CFG_VM_ENABLE
  // Host-side VM control: device SATP for the walker complex, the
  // device-idle TLB flush broadcast, and first-fault readback.
  void set_mmu_satp(uint64_t value);
  void mmu_clear_fault();
  uint64_t mmu_fault_va() const;
  uint32_t mmu_fault_info() const;
#endif

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  class Core* get_core(uint32_t idx) const;

  // Cache flush walk. ProcessorImpl ticks in level order (L1 in parallel
  // → L2 → L3) to avoid downstream evictions racing the next-level walk.
  // L1 fanout: dcache + icache + socket-local {tcache, rtcache} (forwarded
  // to the sockets) + cluster-local {rcache, ocache}.
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

#ifdef VX_CFG_EXT_RASTER_ENABLE
  // Cluster-shared raster engine (armed by the KMU's delegated draw launch;
  // the per-core SFU wave-pull launches covered-quad waves autonomously).
  RasterCore::Ptr& raster_core();
#endif

protected:
  void on_reset();

private:
  void on_gbar_arrive(const GbarArrive& msg);

  uint32_t       cluster_id_;
  ProcessorImpl* processor_;

  class Impl;
  Impl* impl_;

  friend class SimObject<Cluster>;
};

} // namespace vortex
