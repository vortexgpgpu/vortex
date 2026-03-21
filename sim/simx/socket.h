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
#include "arch.h"
#include "cache_cluster.h"
#include "local_mem.h"
#include "core.h"
#include "constants.h"

namespace vortex {

class Cluster;

class Socket : public SimObject<Socket> {
public:
  struct PerfStats {
    CacheSim::PerfStats icache;
    CacheSim::PerfStats dcache;
  };

  std::vector<SimChannel<MemReq>> mem_req_out;
  std::vector<SimChannel<MemRsp>> mem_rsp_in;

  Socket(const SimContext& ctx,
         const char* name,
         uint32_t socket_id,
         Cluster* cluster,
         const Arch &arch);

  ~Socket();

  uint32_t id() const {
    return socket_id_;
  }

  Cluster* cluster() const {
    return cluster_;
  }

  void reset();

  void tick();

  void attach_ram(RAM* ram);

#ifdef VM_ENABLE
  void set_satp(uint64_t satp);
#endif

  bool running() const;

  int get_exitcode() const;

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id);

  void global_barrier_resume(uint32_t bar_id, uint32_t core_id);

  PerfStats perf_stats() const;

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  Core::Ptr& core(uint32_t idx) { return cores_.at(idx); }

private:
  uint32_t                socket_id_;
  Cluster*                cluster_;
  std::vector<Core::Ptr>  cores_;
  CacheCluster::Ptr       icaches_;
  CacheCluster::Ptr       dcaches_;
};

} // namespace vortex
