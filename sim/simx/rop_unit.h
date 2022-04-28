#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"
#include "types.h"
namespace vortex {

class RAM;
class Core;

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
  };

  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    std::vector<mem_addr_size_t> mem_rd_addrs;
    std::vector<mem_addr_size_t> mem_wr_addrs;
    uint32_t core_id;
    uint32_t uuid;
  };

  class DCRS {
  private:
    std::array<uint32_t, DCR_ROP_STATE_COUNT> states_;

  public:
    DCRS() {
      this->clear();
    }

    void clear() {
      for (auto& state : states_) {
        state = 0;
      }
    }

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

  SimPort<TraceData::Ptr> Input;

  RopUnit(const SimContext& ctx, 
          const char* name,  
          const Arch &arch, 
          const DCRS& dcrs);    

  ~RopUnit();

  void attach_ram(RAM* mem);

  void reset();

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, TraceData::Ptr trace_data);

  void tick();

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}