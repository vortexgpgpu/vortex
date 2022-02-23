#include "rasterunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RasterUnit::Impl {
private:
    RasterUnit* simobject_;    
    Core* core_;
    PerfStats perf_stats_;

public:
    Impl(RasterUnit* simobject, Core* core) 
      : simobject_(simobject)
      , core_(core)
    {}

    ~Impl() {}

    void clear() {
        //--
    }

    bool pop(raster_quad_t* quad) {
        __unused (quad);
        return false;
    }

    void tick() {
        //--
    }

    const PerfStats& perf_stats() const { 
        return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RasterUnit::RasterUnit(const SimContext& ctx, const char* name, Core* core) 
  : SimObject<RasterUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core)) 
{}

RasterUnit::~RasterUnit() {
  delete impl_;
}

void RasterUnit::reset() {
  impl_->clear();
}

bool RasterUnit::pop(raster_quad_t* quad) {
  return impl_->pop(quad);
}

void RasterUnit::tick() {
  impl_->tick();
}

const RasterUnit::PerfStats& RasterUnit::perf_stats() const {
    return impl_->perf_stats();
}