#pragma once

#include <simobject.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "types.h"
#include "pipeline.h"

namespace vortex {

class RAM;

class RasterUnit : public SimObject<RasterUnit> {
public:
  using fixed16_t = cocogfx::TFixed<16>;
  using vec3_fx_t = cocogfx::TVector3<fixed16_t>;

  struct Stamp {
    uint32_t  x;
    uint32_t  y;  
    uint32_t  mask;
    std::array<vec3_fx_t, 4> bcoords; // barycentric coordinates        
    uint32_t  pid;

    Stamp* next_;
    Stamp* prev_;

    Stamp(uint32_t x, uint32_t y, uint32_t mask, const std::array<vec3_fx_t, 4>& bcoords, uint32_t  pid) 
      : x(x)
      , y(y)
      , mask(mask)
      , bcoords(bcoords)
      , pid(pid)
      , next_(nullptr)
      , prev_(nullptr) 
    {}

    void* operator new(size_t /*size*/) {
      return allocator().allocate();
    }

    void operator delete(void* ptr) {
      allocator().deallocate(ptr);
    }

  private:

    static MemoryPool<Stamp>& allocator() {
      static MemoryPool<Stamp> instance(1024);
      return instance;
    }
  };

  struct Config {
    uint32_t tile_logsize;
    uint32_t block_logsize;
  };
  
  struct PerfStats {        
    uint64_t reads;
    uint64_t latency;
    uint64_t stalls;

    PerfStats() 
      : reads(0)
      , latency(0)
      , stalls(0)
    {}
    
    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads   += rhs.reads;
      this->latency += rhs.latency;
      this->stalls  += rhs.stalls;
      return *this;
    }
  };
  
  class DCRS {
  private:
    std::array<uint32_t, DCR_RASTER_STATE_COUNT> states_;

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
      uint32_t state = DCR_RASTER_STATE(addr);
      return states_.at(state);
    }

    void write(uint32_t addr, uint32_t value) {
      uint32_t state = DCR_RASTER_STATE(addr);
      states_.at(state) = value;
    }
  };
  
  SimPort<MemReq> MemReqs;
  SimPort<MemRsp> MemRsps;

  SimPort<pipeline_trace_t*> Input;
  SimPort<pipeline_trace_t*> Output;
  
  RasterUnit(const SimContext& ctx, 
            const char* name,
            uint32_t index,
            uint32_t cores_per_unit,
            const Arch &arch, 
            const DCRS& dcrs,            
            const Config& config);    

  ~RasterUnit();

  void reset();

  void tick();

  uint32_t id() const;

  void attach_ram(RAM* mem); 

  uint32_t csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr);

  void csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

  uint32_t fetch(uint32_t cid, uint32_t wid, uint32_t tid);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}