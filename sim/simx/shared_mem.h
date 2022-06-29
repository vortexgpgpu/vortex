#pragma once

#include <simobject.h>
#include "types.h"

namespace vortex {

class SharedMem : public SimObject<SharedMem> {
public:
  struct Config {
    uint32_t capacity;
    uint32_t line_size;
    uint32_t num_reqs;
    uint32_t num_banks; 
    uint32_t bank_offset;
    uint32_t latency;
    bool write_reponse;
  };

  struct PerfStats {
    uint64_t reads;
    uint64_t writes;
    uint64_t bank_stalls;

    PerfStats() 
      : reads(0)
      , writes(0)
      , bank_stalls(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads += rhs.reads;
      this->writes += rhs.writes;
      this->bank_stalls += rhs.bank_stalls;
      return *this;
    }
  };

  std::vector<SimPort<MemReq>> Inputs;
  std::vector<SimPort<MemRsp>> Outputs;

  SharedMem(const SimContext& ctx, const char* name, const Config& config);    
  virtual ~SharedMem();

  void reset();

  void read(void* data, uint64_t addr, uint32_t size);

  void write(const void* data, uint64_t addr, uint32_t size);

  void tick();

  const PerfStats& perf_stats() const;

protected:
  class Impl;
  Impl* impl_;
};

}
