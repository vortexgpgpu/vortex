#include "rasterunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RasterUnit::Impl {
private:
    std::array<uint32_t, NUM_RASTER_STATES> states_;
    RasterUnit* simobject_;    
    Core* core_;
    PerfStats perf_stats_;

public:
    Impl(RasterUnit* simobject, Core* core) 
      : simobject_(simobject)
      , core_(core)
    {}

    ~Impl() {}

    void reset() {
        for (auto& state : states_) {
            state = 0;
        }
    }

    uint32_t csr_read(uint32_t addr) {
        uint32_t state = CSR_RASTER_STATE(addr);
        return states_.at(state);
    }
  
    void csr_write(uint32_t addr, uint32_t value) {
        uint32_t state = CSR_RASTER_STATE(addr);
        states_.at(state) = value;
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
  impl_->reset();
}

uint32_t RasterUnit::csr_read(uint32_t addr) {
  return impl_->csr_read(addr);
}

void RasterUnit::csr_write(uint32_t addr, uint32_t value) {
  impl_->csr_write(addr, value);
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