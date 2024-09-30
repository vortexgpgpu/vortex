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

#include "mem_sim.h"
#include "cache_sim.h"
#include "constants.h"
#include "dcrs.h"
#include "cluster.h"

namespace vortex {

class ProcessorImpl {
public:
  struct PerfStats {
    CacheSim::PerfStats l3cache;
    MemSim::PerfStats memsim;
    uint64_t mem_reads;
    uint64_t mem_writes;
    uint64_t mem_latency;
  };

  ProcessorImpl(const Arch& arch);
  ~ProcessorImpl();

  void attach_ram(RAM* mem);

  void run();

  void dcr_write(uint32_t addr, uint32_t value);

  PerfStats perf_stats() const;

private:

  void reset();

  const Arch& arch_;
  std::vector<std::shared_ptr<Cluster>> clusters_;
  DCRS dcrs_;
  MemSim::Ptr memsim_;
  CacheSim::Ptr l3cache_;
  uint64_t perf_mem_reads_;
  uint64_t perf_mem_writes_;
  uint64_t perf_mem_latency_;
  uint64_t perf_mem_pending_reads_;
};

}
