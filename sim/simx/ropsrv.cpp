#include "ropsrv.h"
#include "ropunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RopSrv::Impl {
private:
    RopSrv* simobject_;
    Core* core_;    
    const ArchDef& arch_;    
    RopUnit::Ptr rop_unit_;
    PerfStats perf_stats_;

public:
    Impl(RopSrv* simobject,      
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

    void write(uint32_t x, uint32_t y , uint32_t mask, uint32_t color, uint32_t depth) {    
      rop_unit_->write(x, y, mask, 0, color, depth);
    }

    void tick() {
      //--
    }    

    const PerfStats& perf_stats() const { 
      return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RopSrv::RopSrv(const SimContext& ctx, 
               const char* name,  
               Core* core,
               RopUnit::Ptr rop_unit) 
  : SimObject<RopSrv>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core, rop_unit)) 
{}

RopSrv::~RopSrv() {
  delete impl_;
}

void RopSrv::reset() {
  impl_->clear();
}

uint32_t RopSrv::csr_read(uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(wid, tid, addr);
}

void RopSrv::csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(wid, tid, addr, value);
}

void RopSrv::write(uint32_t x, uint32_t y, uint32_t mask, uint32_t color, uint32_t depth) {
  impl_->write(x, y, mask, color, depth);
}

void RopSrv::tick() {
  impl_->tick();
}

const RopSrv::PerfStats& RopSrv::perf_stats() const {
  return impl_->perf_stats();
}