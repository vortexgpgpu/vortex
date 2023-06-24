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
    uint64_t mem_reads;
    uint64_t mem_writes;
    uint64_t mem_latency;
    CacheSim::PerfStats l3cache;
    Cluster::PerfStats clusters;

    PerfStats()
      : mem_reads(0)
      , mem_writes(0)
      , mem_latency(0)
    {}
  };

  ProcessorImpl(const Arch& arch);
  ~ProcessorImpl();

  void attach_ram(RAM* mem);

  int run(bool riscv_test);

  void write_dcr(uint32_t addr, uint32_t value);

  ProcessorImpl::PerfStats perf_stats() const;

private:
 
  void reset();

  const Arch& arch_;
  std::vector<std::shared_ptr<Cluster>> clusters_;
  DCRS dcrs_;
  MemSim::Ptr   memsim_;
  CacheSim::Ptr l3cache_;
  uint64_t perf_mem_reads_;
  uint64_t perf_mem_writes_;
  uint64_t perf_mem_latency_;
  uint64_t perf_mem_pending_reads_;
};

}
