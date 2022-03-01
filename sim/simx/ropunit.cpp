#include "ropunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RopUnit::Impl {
private:
    RopUnit* simobject_;    
    Core* core_;
    PerfStats perf_stats_;

public:
    Impl(RopUnit* simobject, Core* core) 
      : simobject_(simobject)
      , core_(core)
    {
        this->clear();
    }

    ~Impl() {}

    void clear() {
      //--
    }

    void write(uint32_t x, uint32_t y, uint32_t z, uint32_t color) {
        __unused (x);
        __unused (y);
        __unused (z);
        __unused (color);
    }

    void tick() {
        //--
    }    

    const PerfStats& perf_stats() const { 
        return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RopUnit::RopUnit(const SimContext& ctx, const char* name, Core* core) 
  : SimObject<RopUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core)) 
{}

RopUnit::~RopUnit() {
  delete impl_;
}

void RopUnit::reset() {
  impl_->clear();
}

void RopUnit::write(uint32_t x, uint32_t y, uint32_t z, uint32_t color) {
  impl_->write(x, y, z, color);
}

void RopUnit::tick() {
  impl_->tick();
}

const RopUnit::PerfStats& RopUnit::perf_stats() const {
    return impl_->perf_stats();
}