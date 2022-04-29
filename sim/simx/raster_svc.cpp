#include "raster_svc.h"
#include "raster_unit.h"
#include "core.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

using namespace vortex;

using fixed24_t = cocogfx::TFixed<24>;

using vec2_fx2_t = cocogfx::TVector2<fixed24_t>;

class RasterSvc::Impl {
private:

  class CSR {
    private:
      RasterUnit::Stamp *stamp_;

    public:

      CSR() : stamp_(nullptr) { this->clear(); }    
      ~CSR() { this->clear(); }

      void clear() {
        if (stamp_) {
          delete stamp_;
          stamp_ = nullptr;
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

    RasterSvc* simobject_;  
    Core* core_;      
    const Arch& arch_;
    RasterUnit::Ptr raster_unit_;    
    std::vector<CSR> csrs_;
    uint32_t tick_stamps_;
    uint64_t last_pop_time_;

public:
    Impl(RasterSvc* simobject,     
         Core* core,
         RasterUnit::Ptr raster_unit) 
      : simobject_(simobject)
      , core_(core)
      , arch_(core->arch())
      , raster_unit_(raster_unit)
      , csrs_(core->arch().num_warps() * core->arch().num_threads())
      , tick_stamps_(0)
    {}

    ~Impl() {}

    void clear() {
      for (auto& csr : csrs_) {
        csr.clear();
      }
      tick_stamps_ = 0;
      last_pop_time_ = 0;
    } 

    uint32_t fetch(uint32_t wid, uint32_t tid) {      
      auto stamp = raster_unit_->fetch();
      if (nullptr == stamp)
        return 0;      
      uint32_t ltid = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(ltid);
      csr.set_stamp(stamp);
      DT(2, "raster-svc: wid=" << std::dec << wid << ", tid=" << tid << ", x=" << stamp->x << ", y=" << stamp->y << ", mask=" << stamp->mask << ", pid=" << stamp->pid << ", bcoords={" 
        << stamp->bcoords[0].x.data() << " " << stamp->bcoords[1].x.data() << " " << stamp->bcoords[2].x.data() << " " << stamp->bcoords[3].x.data() << ", " 
        << stamp->bcoords[0].y.data() << " " << stamp->bcoords[1].y.data() << " " << stamp->bcoords[2].y.data() << " " << stamp->bcoords[3].y.data() << ", "
        << stamp->bcoords[0].z.data() << " " << stamp->bcoords[1].z.data() << " " << stamp->bcoords[2].z.data() << " " << stamp->bcoords[3].z.data() << "}");
      return (stamp->pid << 1) | 1;
    }

    uint32_t csr_read(int32_t wid, uint32_t tid, uint32_t addr) {
      uint32_t index = wid * arch_.num_threads() + tid;
      auto& csr = csrs_.at(index);
      auto stamp = csr.get_stamp();
      switch (addr) {
      case CSR_RASTER_POS_MASK:
        return (stamp->y << (4 + RASTER_DIM_BITS-1)) | (stamp->x << 4) | stamp->mask;      
      case CSR_RASTER_BCOORD_X0:
      case CSR_RASTER_BCOORD_X1:
      case CSR_RASTER_BCOORD_X2:
      case CSR_RASTER_BCOORD_X3:
        return stamp->bcoords.at(addr - CSR_RASTER_BCOORD_X0).x.data();
      case CSR_RASTER_BCOORD_Y0:
      case CSR_RASTER_BCOORD_Y1:
      case CSR_RASTER_BCOORD_Y2:
      case CSR_RASTER_BCOORD_Y3:
        return stamp->bcoords.at(addr - CSR_RASTER_BCOORD_Y0).y.data();
      case CSR_RASTER_BCOORD_Z0:
      case CSR_RASTER_BCOORD_Z1:
      case CSR_RASTER_BCOORD_Z2:
      case CSR_RASTER_BCOORD_Z3:
        return stamp->bcoords.at(addr - CSR_RASTER_BCOORD_Z0).z.data();
      default:
        std::abort();
      }
      return 0;
    }

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
      __unused (wid);
      __unused (tid);
      __unused (addr);
      __unused (value);
    }

    void tick() {
      // check input queue
      if (simobject_->Input.empty())
        return;

      auto trace = simobject_->Input.front();

      if (!raster_unit_->done()) {
        auto num_threads = trace->tmask.count();

        while (tick_stamps_ < num_threads) {
          if (raster_unit_->Output.empty())
            return;       

          auto& stampRsp = raster_unit_->Output.front();
          tick_stamps_ += stampRsp.count;
          raster_unit_->Output.pop();
        }
        
        tick_stamps_ -= num_threads;
      }
            
      simobject_->Output.send(trace, 1);

      simobject_->Input.pop();
    }
};

///////////////////////////////////////////////////////////////////////////////

RasterSvc::RasterSvc(const SimContext& ctx, 
                     const char* name,  
                     Core* core,
                     RasterUnit::Ptr raster_unit)  
: SimObject<RasterSvc>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, core, raster_unit)) 
{}

RasterSvc::~RasterSvc() {
  delete impl_;
}

void RasterSvc::reset() {
  impl_->clear();
}

uint32_t RasterSvc::csr_read(uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(wid, tid, addr);
}

void RasterSvc::csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(wid, tid, addr, value);
}

uint32_t RasterSvc::fetch(uint32_t wid, uint32_t tid) {
  return impl_->fetch(wid, tid);
}

void RasterSvc::tick() {
  impl_->tick();
}