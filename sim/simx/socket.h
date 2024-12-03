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
#include "dcrs.h"
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

  std::vector<SimPort<MemReq>> mem_req_ports;
  std::vector<SimPort<MemRsp>> mem_rsp_ports;

  Socket(const SimContext& ctx,
         uint32_t socket_id,
         Cluster* cluster,
         const Arch &arch,
         const DCRS &dcrs);

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

  void barrier(uint32_t bar_id, uint32_t count, uint32_t core_id);

  void resume(uint32_t core_id);

  PerfStats perf_stats() const;

private:
  uint32_t                socket_id_;
  Cluster*                cluster_;
  std::vector<Core::Ptr>  cores_;
  CacheCluster::Ptr       icaches_;
  CacheCluster::Ptr       dcaches_;
};

} // namespace vortex