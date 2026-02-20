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

  struct AsyncClusterBarrier {
  static constexpr uint32_t MAX_CORES = 32;

  CoreMask   arrived_cores;
  CoreMask   waiting_cores;
  uint32_t   arrived_count;
  uint32_t   expect_cores;
  uint32_t   generation;
  std::array<uint32_t, MAX_CORES> wait_phase;

  AsyncClusterBarrier()
    : arrived_count(0)
    , expect_cores(0)
    , generation(0)
  {
    arrived_cores.reset();
    waiting_cores.reset();
    wait_phase.fill(0);
  }

  void reset_all() {
    arrived_cores.reset();
    waiting_cores.reset();
    arrived_count = 0;
    expect_cores  = 0;
    generation    = 0;
    wait_phase.fill(0);
  }
};

  std::vector<SimChannel<MemReq>> mem_req_out;
  std::vector<SimChannel<MemRsp>> mem_rsp_in;

  Cluster(const SimContext& ctx,
          const char* name,
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

  uint32_t get_barrier_phase(uint32_t bar_id) const;

  void barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id);

  PerfStats perf_stats() const;

private:
  uint32_t                    cluster_id_;
  ProcessorImpl*              processor_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<core_barrier_t> barriers_;
  CacheSim::Ptr               l2cache_;
  uint32_t                    cores_per_socket_;
};

} // namespace vortex
