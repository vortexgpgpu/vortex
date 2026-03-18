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
#include "cluster.h"
#include "kmu.h"

namespace vortex {

class ProcessorImpl {
public:
  struct PerfStats {
    CacheSim::PerfStats l3cache;
    MemSim::PerfStats memsim;
    uint64_t mem_reads = 0;
    uint64_t mem_writes = 0;
    uint64_t mem_latency = 0;
  };

  ProcessorImpl(const Arch& arch);
  ~ProcessorImpl();

  void attach_ram(RAM* mem);

  void reset();

  int run();

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

#ifdef VM_ENABLE
  void set_satp(uint64_t satp);
#endif

  PerfStats perf_stats() const;

  Kmu& kmu() { return kmu_; }

private:

  const Arch& arch_;
  Kmu         kmu_;
  std::vector<std::shared_ptr<Cluster>> clusters_;
  MemSim::Ptr memsim_;
  CacheSim::Ptr l3cache_;
  uint64_t perf_mem_reads_;
  uint64_t perf_mem_writes_;
  uint64_t perf_mem_latency_;
  uint64_t perf_mem_pending_reads_;
};

}
