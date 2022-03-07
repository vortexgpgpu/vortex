#include "rastersrv.h"
#include "rasterunit.h"
#include "core.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

using namespace vortex;

using fixed23_t = cocogfx::TFixed<23>;

using vec2_fx2_t = cocogfx::TVector2<fixed23_t>;

struct csr_t {
  uint32_t frag;
  RasterUnit::Stamp *stamp;
  vec2_fx2_t gradients[4];

  csr_t(RasterUnit::Stamp *stamp = nullptr) 
    : stamp(stamp) 
  {}
  
  ~csr_t() { 
    delete stamp; 
  }
};

class RasterSrv::Impl {
private:
    RasterSrv* simobject_;  
    Core* core_;      
    const ArchDef& arch_;
    RasterUnit::Ptr raster_unit_;    
    std::vector<csr_t> csrs_;
    PerfStats perf_stats_;

public:
    Impl(RasterSrv* simobject,     
         Core* core,
         RasterUnit::Ptr raster_unit) 
      : simobject_(simobject)
      , core_(core)
      , arch_(core->arch())
      , raster_unit_(raster_unit)
      , csrs_(core->arch().num_cores() * core->arch().num_warps() * core->arch().num_threads())
    {}

    ~Impl() {}

    void clear() {
      //--
    } 

    uint32_t csr_read(int32_t wid, uint32_t tid, uint32_t addr) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      switch (addr) {
      case CSR_RASTER_X_Y:
        return (csr.stamp->y << 16) | csr.stamp->x;
      case CSR_RASTER_MASK_PID:
        return (csr.stamp->mask << 4) | csr.stamp->pid;
      case CSR_RASTER_FRAG:
        return csr.frag;
      case CSR_RASTER_BCOORD_X:
        return csr.stamp->bcoords[csr.frag].x.data();
      case CSR_RASTER_BCOORD_Y:
        return csr.stamp->bcoords[csr.frag].y.data();
      case CSR_RASTER_BCOORD_Z:
        return csr.stamp->bcoords[csr.frag].z.data();
      case CSR_RASTER_GRAD_X:
        return csr.gradients[csr.frag].x.data();
      case CSR_RASTER_GRAD_Y:
        return csr.gradients[csr.frag].y.data();
      default:
        std::abort();
      }
      return 0;
    }

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      switch (addr) {
      case CSR_RASTER_FRAG:
        csr.frag = value;
        break;
      case CSR_RASTER_GRAD_X:
        csr.gradients[csr.frag].x = fixed23_t::make(value);
        break;
      case CSR_RASTER_GRAD_Y:
        csr.gradients[csr.frag].y = fixed23_t::make(value);
        break;
      default:
        std::abort();
      }
    }

    uint32_t fetch(uint32_t wid, uint32_t tid) {      
      auto stamp = raster_unit_->fetch();
      if (nullptr == stamp)
        return 0;      
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      if (csr.stamp) {
        delete csr.stamp;
      }
      csr.stamp = stamp;
      return (stamp->pid << 1) | 1;
    }

    int32_t interpolate(uint32_t wid, uint32_t tid, 
                        uint32_t quad, int32_t a, int32_t b, int32_t c) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      auto afx = fixed23_t::make(a);
      auto bfx = fixed23_t::make(b);
      auto cfx = fixed23_t::make(c);
      auto out = cocogfx::Dot<fixed23_t>(afx, csr.gradients[quad].x, bfx, csr.gradients[quad].y) + cfx;
      return out.data();
    }

    void tick() {
      //--
    }

    const PerfStats& perf_stats() const { 
        return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RasterSrv::RasterSrv(const SimContext& ctx, 
                     const char* name,  
                     Core* core,
                     RasterUnit::Ptr raster_unit)  
: SimObject<RasterSrv>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core, raster_unit)) 
{}

RasterSrv::~RasterSrv() {
  delete impl_;
}

void RasterSrv::reset() {
  impl_->clear();
}

uint32_t RasterSrv::csr_read(uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(wid, tid, addr);
}

void RasterSrv::csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(wid, tid, addr, value);
}

uint32_t RasterSrv::fetch(uint32_t wid, uint32_t tid) {
  return impl_->fetch(wid, tid);
}

int32_t RasterSrv::interpolate(uint32_t wid, uint32_t tid, 
                               uint32_t q, int32_t a, int32_t b, int32_t c) {
  return impl_->interpolate(wid, tid, q, a, b, c);
}

void RasterSrv::tick() {
  impl_->tick();
}

const RasterSrv::PerfStats& RasterSrv::perf_stats() const {
  return impl_->perf_stats();
}