#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"
#include "types.h"

namespace vortex {

class RAM;

class RopUnit : public SimObject<RopUnit> {
public:
  struct PerfStats {        
    uint64_t reads;
    uint64_t writes;
    uint64_t latency;
    uint64_t stalls;

    PerfStats() 
      : reads(0)
      , writes(0)
      , latency(0)
      , stalls(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads   += rhs.reads;
      this->writes  += rhs.writes;
      this->latency += rhs.latency;
      this->stalls  += rhs.stalls;
      return *this;
    }
  };

  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    std::vector<mem_addr_size_t> mem_rd_addrs;
    std::vector<mem_addr_size_t> mem_wr_addrs;
    uint32_t cid;
    uint32_t uuid;
  };

  class DCRS {
  private:
    std::array<uint32_t, DCR_ROP_STATE_COUNT> states_;

  public:
    uint32_t read(uint32_t addr) const {
      uint32_t state = DCR_ROP_STATE(addr);
      return states_.at(state);
    }

    void write(uint32_t addr, uint32_t value) {
      uint32_t state = DCR_ROP_STATE(addr);
      states_.at(state) = value;
    }
  };

  std::vector<SimPort<MemReq>> MemReqs;
  std::vector<SimPort<MemRsp>> MemRsps;

  SimPort<pipeline_trace_t*> Input;
  SimPort<pipeline_trace_t*> Output;

  RopUnit(const SimContext& ctx, 
          const char* name,  
          uint32_t cores_per_unit,
          const Arch &arch, 
          const DCRS& dcrs);    

  ~RopUnit();

  void reset();

  void tick();

  void attach_ram(RAM* mem);
  
  uint32_t csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr);
  
  void csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

  void write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}