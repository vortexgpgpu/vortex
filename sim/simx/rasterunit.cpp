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

    int32_t interpolate(uint32_t quad, int32_t a, int32_t b, int32_t c) {
      //--
      return 0;
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

int32_t RasterUnit::interpolate(uint32_t quad, int32_t a, int32_t b, int32_t c) {
  return impl_->interpolate(quad, a, b, c);
}

const RasterUnit::PerfStats& RasterUnit::perf_stats() const {
    return impl_->perf_stats();
}