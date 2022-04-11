#include "rop_svc.h"
#include "rop_unit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RopSvc::Impl {
private:
    RopSvc* simobject_;
    Core* core_;    
    const Arch& arch_;    
    RopUnit::Ptr rop_unit_;
    PerfStats perf_stats_;

public:
    Impl(RopSvc* simobject,      
         Core* core,
         RopUnit::Ptr rop_unit) 
      : simobject_(simobject)
      , core_(core)
      , arch_(core->arch())
      , rop_unit_(rop_unit)
    {
        this->clear();
    }

    ~Impl() {}

    void clear() {
      //--
    }

    uint32_t csr_read(uint32_t wid, uint32_t tid, uint32_t addr) {
      //--
      __unused (wid);
      __unused (tid);
      __unused (addr);
      return 0;
    }

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
      //--
      __unused (wid);
      __unused (tid);
      __unused (addr);
      __unused (value);
    }    

    void write(uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth) {
      __unused (wid);
      __unused (tid);
      DT(3, "rop-svc: wid=" << std::dec << wid << ", tid=" << tid << ", x=" << x << ", y=" << y << ", backface=" << is_backface << ", color=0x" << std::hex << color << ", depth=0x" << depth);
      rop_unit_->write(x, y, is_backface, color, depth);
    }

    void tick() {
      // check input queue
      if (simobject_->Input.empty())
          return;

      auto trace = simobject_->Input.front();

      simobject_->Output.send(trace, 1);

      auto time = simobject_->Input.pop();
      perf_stats_.stalls += (SimPlatform::instance().cycles() - time);
    }    

    const PerfStats& perf_stats() const { 
      return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RopSvc::RopSvc(const SimContext& ctx, 
               const char* name,  
               Core* core,
               RopUnit::Ptr rop_unit) 
  : SimObject<RopSvc>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core, rop_unit)) 
{}

RopSvc::~RopSvc() {
  delete impl_;
}

void RopSvc::reset() {
  impl_->clear();
}

uint32_t RopSvc::csr_read(uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(wid, tid, addr);
}

void RopSvc::csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(wid, tid, addr, value);
}

void RopSvc::write(uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth) {
  impl_->write(wid, tid, x, y, is_backface, color, depth);
}

void RopSvc::tick() {
  impl_->tick();
}

const RopSvc::PerfStats& RopSvc::perf_stats() const {
  return impl_->perf_stats();
}