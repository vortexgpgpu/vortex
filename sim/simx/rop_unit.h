#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"
#include "graphics.h"
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
    uint32_t rop_idx;
  };

  using DCRS = graphics::RopDCRS;

  std::vector<SimPort<MemReq>> MemReqs;
  std::vector<SimPort<MemRsp>> MemRsps;

  SimPort<pipeline_trace_t*> Input;
  SimPort<pipeline_trace_t*> Output;

  RopUnit(const SimContext& ctx, 
          const char* name,
          const Arch &arch, 
          const DCRS& dcrs);    

  ~RopUnit();

  void reset();

  void tick();

  void attach_ram(RAM* mem);

  void write(uint32_t cid, uint32_t wid, uint32_t tid, 
             uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, 
             const CSRs& csrs, RopUnit::TraceData::Ptr trace_data);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}