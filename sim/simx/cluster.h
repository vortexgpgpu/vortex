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
#include "socket.h"
#include "constants.h"

namespace vortex {

class ProcessorImpl;

class Cluster : public SimObject<Cluster> {
public:
  struct PerfStats {
    CacheSim::PerfStats l2cache;
  };

  std::vector<SimPort<MemReq>> mem_req_ports;
  std::vector<SimPort<MemRsp>> mem_rsp_ports;

  Cluster(const SimContext& ctx,
          uint32_t cluster_id,
          ProcessorImpl* processor,
          const Arch &arch,
          const DCRS &dcrs);

  ~Cluster();

  uint32_t id() const {
    return cluster_id_;
  }

  ProcessorImpl* processor() const {
    return processor_;
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

  PerfStats perf_stats() const;

private:
  uint32_t                    cluster_id_;
  ProcessorImpl*              processor_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<CoreMask>       barriers_;
  CacheSim::Ptr               l2cache_;
  uint32_t                    cores_per_socket_;
};

} // namespace vortex