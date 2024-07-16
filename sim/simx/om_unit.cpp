// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "om_unit.h"
#include "mem.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <cocogfx/include/color.hpp>
#include <algorithm>

using namespace vortex;

class OutputMerger {
public:
  OutputMerger() {}

  void configure(const graphics::OMDCRS& dcrs) {
    // get device configuration

    depthStencil_.configure(dcrs);
    blender_.configure(dcrs);

    zbuf_baseaddr_ = uint64_t(dcrs.read(VX_DCR_OM_ZBUF_ADDR)) << 6;
    zbuf_pitch_ = dcrs.read(VX_DCR_OM_ZBUF_PITCH);
    depth_writemask_ = dcrs.read(VX_DCR_OM_DEPTH_WRITEMASK) & 0x1;
    stencil_front_writemask_ = dcrs.read(VX_DCR_OM_STENCIL_WRITEMASK) & 0xffff;
    stencil_back_writemask_ = dcrs.read(VX_DCR_OM_STENCIL_WRITEMASK) >> 16;

    cbuf_baseaddr_ = uint64_t(dcrs.read(VX_DCR_OM_CBUF_ADDR)) << 6;
    cbuf_pitch_ = dcrs.read(VX_DCR_OM_CBUF_PITCH);
    auto cbuf_writemask = dcrs.read(VX_DCR_OM_CBUF_WRITEMASK) & 0xf;
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
             OMUnit::TraceData::Ptr trace_data) {
    auto depth_enabled   = depthStencil_.depth_enabled();
    auto stencil_enabled = depthStencil_.stencil_enabled(is_backface);
    auto blend_enabled   = blender_.enabled();

    uint32_t depthstencil;
    uint32_t dst_depthstencil;
    uint32_t dst_color;

    this->read(depth_enabled, stencil_enabled, blend_enabled,
               x, y, &dst_depthstencil, &dst_color, trace_data);

    auto ds_passed = !(depth_enabled || stencil_enabled)
                  || depthStencil_.test(is_backface, depth, dst_depthstencil, &depthstencil);

    if (blend_enabled && ds_passed) {
      color = blender_.blend(color, dst_color);
    }

    this->write(depth_enabled, stencil_enabled, ds_passed, is_backface,
                dst_depthstencil, dst_color, x, y, depthstencil, color, trace_data);
  }

private:

  void read(bool depth_enable,
            bool stencil_enable,
            bool blend_enable,
            uint32_t x,
            uint32_t y,
            uint32_t* depthstencil,
            uint32_t* color,
            OMUnit::TraceData::Ptr trace_data) {
    if (depth_enable || stencil_enable) {
      uint64_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;
      mem_->read(depthstencil, zbuf_addr, 4);
      trace_data->mem_rd_addrs.push_back(zbuf_addr);
      DT(3, "om-depthstencil-read: x=" << std::dec << x << ", y=" << y << ", addr=0x" << std::hex << zbuf_addr << ", depthstencil=0x" << *depthstencil);
    }
    if (color_write_ && (color_read_ || blend_enable)) {
      uint64_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->read(color, cbuf_addr, 4);
      trace_data->mem_rd_addrs.push_back(cbuf_addr);
      DT(3, "om-color-read: x=" << std::dec << x << ", y=" << y << ", addr=0x" << std::hex << cbuf_addr << ", color=0x" << *color);
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
             OMUnit::TraceData::Ptr trace_data) {
    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    auto ds_writeMask = ((depth_enable && ds_passed && depth_writemask_) ? VX_OM_DEPTH_MASK : 0)
                      | (stencil_enable ? (stencil_writemask << VX_OM_DEPTH_BITS) : 0);
    if (ds_writeMask != 0) {
      uint32_t write_value = (dst_depthstencil & ~ds_writeMask) | (depthstencil & ds_writeMask);
      uint64_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;
      mem_->write(&write_value, zbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back(zbuf_addr);
      DT(3, "om-depthstencil-write: x=" << std::dec << x << ", y=" << y << ", addr=0x" << std::hex << zbuf_addr << ", depthstencil=0x" << write_value);
    }

    if (color_write_ && ds_passed) {
      uint32_t write_value = (dst_color & ~cbuf_writemask_) | (color & cbuf_writemask_);
      uint64_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->write(&write_value, cbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back(cbuf_addr);
      DT(3, "om-color-write: x=" << std::dec << x << ", y=" << y << ", addr=0x" << std::hex << cbuf_addr << ", color=0x" << write_value);
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

class OMUnit::Impl {
public:

  Impl(OMUnit* simobject,
       const Arch &arch,
       const DCRS& dcrs)
    : simobject_(simobject)
    , arch_(arch)
    , dcrs_(dcrs)
    , pending_reqs_(OM_MEM_QUEUE_SIZE)
  {
    this->reset();
  }

  ~Impl() {}

  void reset() {
    render_output_.configure(dcrs_);
    last_pop_time_= 0;
    pending_reqs_.clear();
    perf_stats_ = PerfStats();
  }

  void tick() {
    // handle memory response
    for (auto& port : simobject_->MemRsps) {
      if (port.empty())
        continue;
      auto& mem_rsp = port.front();
      auto& entry = pending_reqs_.at(mem_rsp.tag);
      assert(entry.count != 0);
      --entry.count; // track remaining addresses
      if (0 == entry.count) {
        for (uint32_t i = 0, n = entry.data->mem_wr_addrs.size(); i < n; ++i) {
          MemReq mem_req;
          mem_req.addr  = entry.data->mem_wr_addrs.at(i);
          mem_req.write = true;
          mem_req.tag   = mem_rsp.tag;
          mem_req.cid   = mem_rsp.cid;
          mem_req.uuid  = mem_rsp.uuid;
          uint32_t port = i % simobject_->MemReqs.size();
          simobject_->MemReqs.at(port).push(mem_req, 2);
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

    auto trace = simobject_->Input.front();

    // check pending queue capacity
    if (pending_reqs_.full()) {
        if (!trace->log_once(true)) {
            DT(3, "*** " << simobject_->name() << "-om-queue-stall: " << *trace);
        }
        ++perf_stats_.stalls;
        return;
    } else {
        trace->log_once(false);
    }

    auto data = std::dynamic_pointer_cast<OMUnit::TraceData>(trace->data);
    auto tag = pending_reqs_.allocate({data, (uint32_t)data->mem_rd_addrs.size()});

    // schedule read requests first
    for (uint32_t i = 0, n = data->mem_rd_addrs.size(); i < n; ++i) {
      MemReq mem_req;
      mem_req.addr  = data->mem_rd_addrs.at(i);
      mem_req.write = false;
      mem_req.tag   = tag;
      mem_req.cid   = trace->cid;
      mem_req.uuid  = trace->uuid;
      uint32_t port = i % simobject_->MemReqs.size();
      simobject_->MemReqs.at(port).push(mem_req, 2);
      ++perf_stats_.reads;
    }

    if (data->mem_rd_addrs.empty()) {
      // schedule write-only requests
      for (uint32_t i = 0, n = data->mem_wr_addrs.size(); i < n; ++i) {
        MemReq mem_req;
        mem_req.addr  = data->mem_wr_addrs.at(i);
        mem_req.write = true;
        mem_req.tag   = tag;
        mem_req.cid   = trace->cid;
        mem_req.uuid  = trace->uuid;
        uint32_t port = i % simobject_->MemReqs.size();
        simobject_->MemReqs.at(port).push(mem_req, 2);
        ++perf_stats_.writes;
      }
      pending_reqs_.release(tag);
    }

    simobject_->Output.push(trace, 1);
    simobject_->Input.pop();
  }

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth,
             OMUnit::TraceData::Ptr trace_data) {
    render_output_.write(x, y, is_backface, color, depth, trace_data);
  }

  void attach_ram(RAM* mem) {
    render_output_.attach_ram(mem);
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  struct pending_req_t {
    TraceData::Ptr data;
    uint32_t count;
  };

  OMUnit*      simobject_;
  const Arch&   arch_;
  const DCRS&   dcrs_;
  std::unordered_map<uint32_t, uint32_t> csrs_;
  PerfStats     perf_stats_;
  OutputMerger  render_output_;
  uint64_t      last_pop_time_;
  HashTable<pending_req_t> pending_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

OMUnit::OMUnit(const SimContext& ctx,
                 const char* name,
                 const Arch &arch,
                 const DCRS& dcrs)
  : SimObject<OMUnit>(ctx, name)
  , MemReqs(NUM_SFU_LANES, this)
  , MemRsps(NUM_SFU_LANES, this)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, arch, dcrs))
{}

OMUnit::~OMUnit() {
  delete impl_;
}

void OMUnit::reset() {
  impl_->reset();
}

void OMUnit::tick() {
  impl_->tick();
}

void OMUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

void OMUnit::write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth,
                   OMUnit::TraceData::Ptr trace_data) {
  impl_->write(x, y, is_backface, color, depth, trace_data);
}

const OMUnit::PerfStats& OMUnit::perf_stats() const {
  return impl_->perf_stats();
}
