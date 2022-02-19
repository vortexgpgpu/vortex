#include "ropunit.h"
#include "core.h"
#include <VX_config.h>

using namespace vortex;

class RopUnit::Impl {
private:
    std::array<uint32_t, NUM_ROP_STATES> states_;
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
        for (auto& state : states_) {
            state = 0;
        }
    }

    uint32_t csr_read(uint32_t addr) {
        uint32_t state = CSR_ROP_STATE(addr);
        return states_.at(state);
    }
  
    void csr_write(uint32_t addr, uint32_t value) {
        uint32_t state = CSR_ROP_STATE(addr);
        states_.at(state) = value;
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
  , impl_(new Impl(this, core)) 
{}

RopUnit::~RopUnit() {
  delete impl_;
}

void RopUnit::reset() {
  impl_->clear();
}

uint32_t RopUnit::csr_read(uint32_t addr) {
  return impl_->csr_read(addr);
}

void RopUnit::csr_write(uint32_t addr, uint32_t value) {
  impl_->csr_write(addr, value);
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