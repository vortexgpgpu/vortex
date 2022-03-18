#include "raster_srv.h"
#include "raster_unit.h"
#include "core.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

using namespace vortex;

using fixed24_t = cocogfx::TFixed<24>;

using vec2_fx2_t = cocogfx::TVector2<fixed24_t>;

class RasterSrv::Impl {
private:

  class CSR {
    private:
      RasterUnit::Stamp *stamp_;

    public:
      uint32_t                  frag;  
      std::array<vec2_fx2_t, 4> gradients;

      CSR() : stamp_(nullptr) { this->clear(); }    
      ~CSR() { this->clear(); }

      void clear() {
        if (stamp_) {
          delete stamp_;
          stamp_ = nullptr;
        }
        frag = 0;
        for (auto& grad : gradients) {
          grad.x = fixed24_t(0);
          grad.y = fixed24_t(0);
        }
      }

      void set_stamp(RasterUnit::Stamp *stamp) {
        if (stamp_)
          delete stamp_;
        stamp_ = stamp;
      }

      RasterUnit::Stamp* get_stamp() const {
        assert(stamp_);
        return stamp_;
      }
    };

    RasterSrv* simobject_;  
    Core* core_;      
    const Arch& arch_;
    RasterUnit::Ptr raster_unit_;    
    std::vector<CSR> csrs_;
    PerfStats perf_stats_;

public:
    Impl(RasterSrv* simobject,     
         Core* core,
         RasterUnit::Ptr raster_unit) 
      : simobject_(simobject)
      , core_(core)
      , arch_(core->arch())
      , raster_unit_(raster_unit)
      , csrs_(core->arch().num_warps() * core->arch().num_threads())
    {}

    ~Impl() {}

    void clear() {
      for (auto& csr : csrs_) {
        csr.clear();
      }
    } 

    uint32_t fetch(uint32_t wid, uint32_t tid) {      
      auto stamp = raster_unit_->fetch();
      if (nullptr == stamp)
        return 0;      
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      csr.set_stamp(stamp);
      return (stamp->pid << 1) | 1;
    }

    uint32_t csr_read(int32_t wid, uint32_t tid, uint32_t addr) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      auto stamp = csr.get_stamp();
      if (0 == (stamp->mask & (1 << csr.frag)) 
       && addr != CSR_RASTER_FRAG)
        return 0;
      auto i  = csr.frag & 0x1;
      auto j  = csr.frag >> 1;
      auto px = stamp->x + i;
      auto py = stamp->y + j;
      __unused (px);
      __unused (py);
      switch (addr) {
      case CSR_RASTER_FRAG:
        return csr.frag;
      case CSR_RASTER_X_Y:
        return (stamp->y << 16) | stamp->x;
      case CSR_RASTER_MASK_PID:
        return (stamp->pid << 4) | stamp->mask;      
      case CSR_RASTER_BCOORD_X:
        //printf("bcoord.x[%d,%d]=%d\n", px, py, stamp->bcoords.at(csr.frag).x.data());
        return stamp->bcoords.at(csr.frag).x.data();
      case CSR_RASTER_BCOORD_Y:
        //printf("bcoord.y[%d,%d]=%d\n", px, py, stamp->bcoords.at(csr.frag).y.data());
        return stamp->bcoords.at(csr.frag).y.data();
      case CSR_RASTER_BCOORD_Z:
        //printf("bcoord.z[%d,%d]=%d\n", px, py, stamp->bcoords.at(csr.frag).z.data());
        return stamp->bcoords.at(csr.frag).z.data();
      case CSR_RASTER_GRAD_X:
        return csr.gradients.at(csr.frag).x.data();
      case CSR_RASTER_GRAD_Y:
        return csr.gradients.at(csr.frag).y.data();
      default:
        std::abort();
      }
      return 0;
    }

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      auto stamp = csr.get_stamp();
      if (0 == (stamp->mask & (1 << csr.frag)) 
       && addr != CSR_RASTER_FRAG)
        return;
      auto i  = csr.frag & 0x1;
      auto j  = csr.frag >> 1;
      auto px = stamp->x + i;
      auto py = stamp->y + j;
      __unused (px);
      __unused (py);
      switch (addr) {
      case CSR_RASTER_FRAG:
        csr.frag = value;
        break;
      case CSR_RASTER_GRAD_X:
        csr.gradients.at(csr.frag).x = fixed24_t::make(value);
        //printf("grad.x[%d,%d]=%d\n", px, py, csr.gradients.at(csr.frag).x.data());
        break;
      case CSR_RASTER_GRAD_Y:
        csr.gradients.at(csr.frag).y = fixed24_t::make(value);
        //printf("grad.y[%d,%d]=%d\n", px, py, csr.gradients.at(csr.frag).y.data());
        break;
      default:
        std::abort();
      }
    }

    int32_t interpolate(uint32_t wid, uint32_t tid, int32_t a, int32_t b, int32_t c) {
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      auto stamp = csr.get_stamp();
      if (0 == (stamp->mask & (1 << csr.frag)))
        return 0;
      auto i   = csr.frag & 0x1;
      auto j   = csr.frag >> 1;
      auto px  = stamp->x + i;
      auto py  = stamp->y + j;
      __unused (px);
      __unused (py);
      auto ax  = fixed24_t::make(a);
      auto bx  = fixed24_t::make(b);
      auto cx  = fixed24_t::make(c);
      auto out = cocogfx::Dot<fixed24_t>(ax, csr.gradients.at(csr.frag).x, bx, csr.gradients.at(csr.frag).y) + cx;
      //printf("interpolate[%d,%d](a=%d, b=%d, c=%d)=%d\n", px, py, a, b, c, out.data());
      return out.data();
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

int32_t RasterSrv::interpolate(uint32_t wid, uint32_t tid, int32_t a, int32_t b, int32_t c) {
  return impl_->interpolate(wid, tid, a, b, c);
}

void RasterSrv::tick() {
  impl_->tick();
}

const RasterSrv::PerfStats& RasterSrv::perf_stats() const {
  return impl_->perf_stats();
}