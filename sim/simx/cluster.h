#pragma once

#include <simobject.h>
#include "dcrs.h"
#include "arch.h"
#include "cache_cluster.h"
#include "shared_mem.h"
#include "raster_unit.h"
#include "rop_unit.h"
#include "tex_unit.h"
#include "core.h"
#include "constants.h"

namespace vortex {

class ProcessorImpl;

class Cluster {
public:
  struct PerfStats {
    RasterUnit::PerfStats raster_unit;
    RopUnit::PerfStats    rop_unit;
    TexUnit::PerfStats    tex_unit;
    CacheSim::PerfStats   icache;
    CacheSim::PerfStats   dcache;
    SharedMem::PerfStats  sharedmem;
    CacheSim::PerfStats   l2cache;
    CacheSim::PerfStats   tcache;
    CacheSim::PerfStats   ocache;
    CacheSim::PerfStats   rcache;

    PerfStats& operator+=(const PerfStats& rhs) {
      this->raster_unit += rhs.raster_unit;
      this->rop_unit    += rhs.rop_unit;
      this->tex_unit    += rhs.tex_unit;
      this->icache      += rhs.icache;
      this->dcache      += rhs.dcache;
      this->sharedmem   += rhs.sharedmem;
      this->l2cache     += rhs.l2cache;
      this->tcache      += rhs.tcache;
      this->ocache      += rhs.ocache;
      this->rcache      += rhs.rcache;
      return *this;
    }
  };

  Cluster(uint32_t cluster_id, uint32_t cores_per_cluster, ProcessorImpl* processor, const Arch &arch, const DCRS &dcrs);

  ~Cluster();

  void attach_ram(RAM* ram);

  bool running() const;

  bool getIRegValue(Word* value, int reg) const;

  void bind(SimPort<MemReq>* mem_req_port, SimPort<MemRsp>* mem_rsp_port);

  ProcessorImpl* processor() const;

  Cluster::PerfStats perf_stats() const;
  
private:
  uint32_t                     cluster_id_;  
  std::vector<Core::Ptr>       cores_;
  std::vector<RasterUnit::Ptr> raster_units_;
  std::vector<RopUnit::Ptr>    rop_units_;
  std::vector<TexUnit::Ptr>    tex_units_;
  CacheSim::Ptr                l2cache_;
  CacheCluster::Ptr            icaches_;
  CacheCluster::Ptr            dcaches_;
  std::vector<SharedMem::Ptr>  sharedmems_;
  CacheCluster::Ptr            tcaches_;
  CacheCluster::Ptr            ocaches_;
  CacheCluster::Ptr            rcaches_;
  ProcessorImpl*               processor_;
};

} // namespace vortex