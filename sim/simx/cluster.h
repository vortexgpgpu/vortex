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
#ifdef EXT_DXA_ENABLE
#include "dxa_core.h"
#endif
#ifdef EXT_TEX_ENABLE
#include "tex_core.h"
#endif
#ifdef EXT_OM_ENABLE
#include "om_core.h"
#endif
#ifdef EXT_RASTER_ENABLE
#include "raster_core.h"
#endif

namespace vortex {

class ProcessorImpl;

class Cluster : public SimObject<Cluster> {
public:
  struct PerfStats {
    Cache::PerfStats l2cache;
#ifdef EXT_DXA_ENABLE
    DxaCore::PerfStats dxa;
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

  // Cache flush walk: dcache → l2cache (sequential to avoid evictions
  // racing the L2 walk). Caller ticks the simulator until done.
  void dcache_flush_begin();
  bool dcache_flush_done() const;
  void l2_flush_begin();
  bool l2_flush_done() const;

#ifdef EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core();
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
