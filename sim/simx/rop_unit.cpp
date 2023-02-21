#include "rop_unit.h"
#include "mem.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <cocogfx/include/color.hpp>
#include <algorithm>

using namespace vortex;

class RenderOutput {
public:  
  RenderOutput() {}

  void configure(const graphics::RopDCRS& dcrs) {
    // get device configuration
    
    depthStencil_.configure(dcrs);
    blender_.configure(dcrs);

    zbuf_baseaddr_ = dcrs.read(DCR_ROP_ZBUF_ADDR);
    zbuf_pitch_    = dcrs.read(DCR_ROP_ZBUF_PITCH);
    depth_writemask_ = dcrs.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;
    stencil_front_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) & 0xffff;
    stencil_back_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) >> 16;

    cbuf_baseaddr_ = dcrs.read(DCR_ROP_CBUF_ADDR);
    cbuf_pitch_    = dcrs.read(DCR_ROP_CBUF_PITCH);
    auto cbuf_writemask = dcrs.read(DCR_ROP_CBUF_WRITEMASK) & 0xf;
    cbuf_writemask_ = (((cbuf_writemask >> 0) & 0x1) * 0x000000ff) 
                    | (((cbuf_writemask >> 1) & 0x1) * 0x0000ff00) 
                    | (((cbuf_writemask >> 2) & 0x1) * 0x00ff0000) 
                    | (((cbuf_writemask >> 3) & 0x1) * 0xff000000);
    color_read_  = (cbuf_writemask != 0xf);
    color_write_ = (cbuf_writemask != 0x0);
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  void write(uint32_t x, 
             uint32_t y, 
             bool is_backface, 
             uint32_t color, 
             uint32_t depth, 
             RopUnit::TraceData::Ptr trace_data) {
    auto depth_enabled   = depthStencil_.depth_enabled();
    auto stencil_enabled = depthStencil_.stencil_enabled(is_backface);
    auto blend_enabled   = blender_.enabled();

    uint32_t depthstencil;    
    uint32_t dst_depthstencil;
    uint32_t dst_color;    

    this->read(depth_enabled, stencil_enabled, blend_enabled, x, y, &dst_depthstencil, &dst_color, trace_data);
    
    auto ds_passed = !(depth_enabled || stencil_enabled)
                  || depthStencil_.test(is_backface, depth, dst_depthstencil, &depthstencil);
    
    if (blend_enabled && ds_passed) {
      color = blender_.blend(color, dst_color);
    }
    
    this->write(depth_enabled, stencil_enabled, ds_passed, is_backface, dst_depthstencil, dst_color, x, y, depthstencil, color, trace_data);
  }

private:

  void read(bool depth_enable,
            bool stencil_enable, 
            bool blend_enable,
            uint32_t x, 
            uint32_t y,
            uint32_t* depthstencil,
            uint32_t* color, 
            RopUnit::TraceData::Ptr trace_data) {
    if (depth_enable || stencil_enable) {
      uint32_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;
      mem_->read(depthstencil, zbuf_addr, 4);     
      trace_data->mem_rd_addrs.push_back({zbuf_addr, 4});
    }

    if (color_write_ && (color_read_ || blend_enable)) {
      uint32_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->read(color, cbuf_addr, 4);
      trace_data->mem_rd_addrs.push_back({cbuf_addr, 4});
    }
  }

  void write(bool depth_enable,
             bool stencil_enable, 
             bool ds_passed,
             bool is_backface,
             uint32_t dst_depthstencil,
             uint32_t dst_color,
             uint32_t x, 
             uint32_t y, 
             uint32_t depthstencil, 
             uint32_t color, 
             RopUnit::TraceData::Ptr trace_data) {

    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    auto ds_writeMask = ((depth_enable && ds_passed && depth_writemask_) ? ROP_DEPTH_MASK : 0) 
                      | (stencil_enable ? (stencil_writemask << ROP_DEPTH_BITS) : 0);
    if (ds_writeMask != 0) {      
      uint32_t write_value = (dst_depthstencil & ~ds_writeMask) | (depthstencil & ds_writeMask);
      uint32_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;        
      mem_->write(&write_value, zbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back({zbuf_addr, 4});
      DT(3, "rop-depthstencil: x=" << std::dec << x << ", y=" << y << ", depthstencil=0x" << std::hex << write_value);
    }

    if (color_write_ && ds_passed) {   
      uint32_t write_value = (dst_color & ~cbuf_writemask_) | (color & cbuf_writemask_);
      uint32_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->write(&write_value, cbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back({cbuf_addr, 4});
      DT(3, "rop-color: x=" << std::dec << x << ", y=" << y << ", color=0x" << std::hex << write_value);
    }
  }

  graphics::DepthTencil depthStencil_;
  graphics::Blender blender_;
  RAM* mem_;
  
  uint32_t zbuf_baseaddr_;
  uint32_t zbuf_pitch_;
  bool     depth_writemask_;
  uint32_t stencil_front_writemask_; 
  uint32_t stencil_back_writemask_;

  uint32_t cbuf_baseaddr_;
  uint32_t cbuf_pitch_;
  uint32_t cbuf_writemask_;

  bool color_read_;
  bool color_write_;
};

///////////////////////////////////////////////////////////////////////////////

class RopUnit::Impl {
private:
  struct pending_req_t {
    TraceData::Ptr data;
    uint32_t count;
  };

  RopUnit*      simobject_;    
  const Arch&   arch_;    
  const DCRS&   dcrs_;
  PerfStats     perf_stats_;
  RenderOutput  render_output_;
  HashTable<pending_req_t> pending_reqs_;
  uint64_t      last_pop_time_;

public:
  Impl(RopUnit* simobject,   
       uint32_t cores_per_unit,   
       const Arch &arch,
       const DCRS& dcrs) 
    : simobject_(simobject)
    , arch_(arch)
    , dcrs_(dcrs)
    , pending_reqs_(ROP_MEM_QUEUE_SIZE)
  {
    __unused (cores_per_unit);
    this->reset();
  }

  ~Impl() {}

  void reset() {
    render_output_.configure(dcrs_);
    last_pop_time_= 0;
  }

  void tick() {
    // handle memory response
    for (auto& port : simobject_->MemRsps) {
      if (port.empty())
        continue;
      auto& mem_rsp = port.front();
      auto& entry = pending_reqs_.at(mem_rsp.tag);
      assert(entry.count);
      --entry.count; // track remaining blocks 
      if (0 == entry.count) {
        auto& mem_wr_addrs = entry.data->mem_wr_addrs;
        for (uint32_t i = 0, n = mem_wr_addrs.size(); i < n; ++i) {
          uint32_t j = i % simobject_->MemReqs.size();
          MemReq mem_req;
          mem_req.addr  = mem_wr_addrs.at(i).addr;
          mem_req.write = true;
          mem_req.tag   = 0;
          mem_req.cid   = mem_rsp.cid;
          mem_req.uuid  = mem_rsp.uuid;
          simobject_->MemReqs.at(j).send(mem_req, 2);
          ++perf_stats_.writes;
        }
        pending_reqs_.release(mem_rsp.tag);
      }   
      port.pop();
    }    

    for (int i = 0, n = pending_reqs_.size(); i < n; ++i) {
      if (pending_reqs_.contains(i))
        perf_stats_.latency += pending_reqs_.at(i).count;
    }    

    // check input trace
    if (simobject_->Input.empty())
      return;
      
    perf_stats_.stalls += simobject_->Input.stalled();
    auto trace = simobject_->Input.front();    
    auto data  = std::dynamic_pointer_cast<RopUnit::TraceData>(trace->data);
    data->cid  = trace->cid;
    data->uuid = trace->uuid;
    if (!data->mem_rd_addrs.empty()) {
      if (pending_reqs_.full())
        return;
      auto tag = pending_reqs_.allocate({data, (uint32_t)data->mem_rd_addrs.size()});
      for (uint32_t i = 0, n = data->mem_rd_addrs.size(); i < n; ++i) {
        uint32_t j = i % simobject_->MemReqs.size();
        MemReq mem_req;
        mem_req.addr  = data->mem_rd_addrs.at(i).addr;
        mem_req.write = false;
        mem_req.tag   = tag;
        mem_req.cid   = data->cid;
        mem_req.uuid  = data->uuid;
        simobject_->MemReqs.at(j).send(mem_req, 1);
        ++perf_stats_.reads;
      }
    } else {
      for (uint32_t i = 0, n = data->mem_wr_addrs.size(); i < n; ++i) {
        uint32_t j = i % simobject_->MemReqs.size();
        MemReq mem_req;
        mem_req.addr  = data->mem_wr_addrs.at(i).addr;
        mem_req.write = true;
        mem_req.tag   = 0;
        mem_req.cid   = data->cid;
        mem_req.uuid  = data->uuid;
        simobject_->MemReqs.at(j).send(mem_req, 1);
        ++perf_stats_.writes;
      }
    }

    simobject_->Output.send(trace, 1);
    simobject_->Input.pop();
  }

  void attach_ram(RAM* mem) {
    render_output_.attach_ram(mem);
  }

  uint32_t csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr) {
    __unused (cid, wid, tid, addr);
    return 0;
  }

  void csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
    __unused (cid, wid, tid, addr, value);
  }   

  void write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data) {      
    __unused (cid, wid, tid);    
    DT(2, "rop-write: cid=" << std::dec << cid << ", wid=" << wid << ", tid=" << tid << ", x=" << x << ", y=" << y << ", backface=" << is_backface << ", color=0x" << std::hex << color << ", depth=0x" << depth);
    render_output_.write(x, y, is_backface, color, depth, trace_data);    
  }

  const PerfStats& perf_stats() const { 
    return perf_stats_; 
  }
};

///////////////////////////////////////////////////////////////////////////////

RopUnit::RopUnit(const SimContext& ctx, 
                 const char* name,          
                 uint32_t cores_per_unit,             
                 const Arch &arch, 
                 const DCRS& dcrs) 
  : SimObject<RopUnit>(ctx, name)
  , MemReqs(arch.num_threads(), this)
  , MemRsps(arch.num_threads(), this)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, cores_per_unit, arch, dcrs)) 
{}

RopUnit::~RopUnit() {
  delete impl_;
}

void RopUnit::reset() {
  impl_->reset();
}

void RopUnit::tick() {
  impl_->tick();
}

void RopUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

uint32_t RopUnit::csr_read(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr) {
  return impl_->csr_read(cid, wid, tid, addr);
}

void RopUnit::csr_write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value) {
  impl_->csr_write(cid, wid, tid, addr, value);
}

void RopUnit::write(uint32_t cid, uint32_t wid, uint32_t tid, uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data) {
  impl_->write(cid, wid, tid, x, y, is_backface, color, depth, trace_data);
}

const RopUnit::PerfStats& RopUnit::perf_stats() const {
  return impl_->perf_stats();
}
