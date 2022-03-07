#include "texunit.h"
#include "core.h"
#include <texturing.h>
#include <VX_config.h>

using namespace vortex;
using namespace cocogfx;

class TexUnit::Impl {
private:
    struct pending_req_t {
      pipeline_trace_t* trace;
      uint32_t count;
    };
    TexUnit* simobject_;
    Config config_;
    Core* core_;
    const DCRS& dcrs_;
    CacheSim::Ptr tcache_;
    uint32_t num_threads_;
    HashTable<pending_req_t> pending_reqs_;
    PerfStats perf_stats_;

public:
    Impl(TexUnit* simobject, const Config& config, Core* core) 
      : simobject_(simobject)
      , config_(config)
      , core_(core) 
      , dcrs_(core->dcrs_.tex_dcrs)
      , tcache_(core->tcache_)
      , num_threads_(core->arch().num_threads())
      , pending_reqs_(TEXQ_SIZE)
    {
      this->clear();
    }

    ~Impl() {}

    void clear() {
      pending_reqs_.clear();
    }

    uint32_t read(uint32_t stage, int32_t u, int32_t v, int32_t lod, TraceData* trace_data) {
      auto& states = dcrs_.at(stage);
      auto xu = TFixed<TEX_FXD_FRAC>::make(u);
      auto xv = TFixed<TEX_FXD_FRAC>::make(v);
      auto base_addr  = states.at(TEX_STATE_ADDR) + states.at(TEX_STATE_MIPOFF(lod));
      auto logdim     = states.at(TEX_STATE_LOGDIM);
      auto log_width  = std::max<int32_t>((logdim & 0xffff) - lod, 0);
      auto log_height = std::max<int32_t>((logdim >> 16) - lod, 0);
      auto format     = states.at(TEX_STATE_FORMAT);    
      auto filter     = states.at(TEX_STATE_FILTER);    
      auto wrap       = states.at(TEX_STATE_WRAP);
      auto wrapu      = wrap & 0xffff;
      auto wrapv      = wrap >> 16;

      auto stride = Stride(format);
      
      switch (filter) {
      case TEX_FILTER_BILINEAR: {
        // addressing
        uint32_t offset00, offset01, offset10, offset11;
        uint32_t alpha, beta;
        TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv, 
          &offset00, &offset01, &offset10, &offset11, &alpha, &beta);

        uint32_t addr00 = base_addr + offset00 * stride;
        uint32_t addr01 = base_addr + offset01 * stride;
        uint32_t addr10 = base_addr + offset10 * stride;
        uint32_t addr11 = base_addr + offset11 * stride;

        // memory lookup
        uint32_t texel00(0), texel01(0), texel10(0), texel11(0);
        core_->dcache_read(&texel00, addr00, stride);
        core_->dcache_read(&texel01, addr01, stride);
        core_->dcache_read(&texel10, addr10, stride);
        core_->dcache_read(&texel11, addr11, stride);

        trace_data->mem_addrs.push_back({{addr00, stride}, 
                                         {addr01, stride}, 
                                         {addr10, stride}, 
                                         {addr11, stride}});

        // filtering
        auto color = TexFilterLinear(
          format, texel00, texel01, texel10, texel11, alpha, beta);
        return color;
      }
      case TEX_FILTER_POINT: {
        // addressing
        uint32_t offset;
        TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);
        
        uint32_t addr = base_addr + offset * stride;

        // memory lookup
        uint32_t texel(0);
        core_->dcache_read(&texel, addr, stride);
        trace_data->mem_addrs.push_back({{addr, stride}});

        // filtering
        auto color = TexFilterPoint(format, texel);
        return color;
      }
      default:
        std::abort();
        return 0;
    }
  }

  void tick() {
    // handle memory response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& tcache_rsp_port = tcache_->CoreRspPorts.at(t);
        if (tcache_rsp_port.empty())
            continue;
        auto& mem_rsp = tcache_rsp_port.front();
        auto& entry = pending_reqs_.at(mem_rsp.tag);  
        auto trace = entry.trace;
        DT(3, "tex-rsp: tag=" << mem_rsp.tag << ", tid=" << t << ", " << *trace);  
        assert(entry.count);
        --entry.count; // track remaining blocks 
        if (0 == entry.count) {
            simobject_->Output.send(trace, config_.sampler_latency);
            pending_reqs_.release(mem_rsp.tag);
        }   
        tcache_rsp_port.pop();
    }

    // check input queue
    if (simobject_->Input.empty())
        return;

    auto trace = simobject_->Input.front();

    // check pending queue capacity    
    if (pending_reqs_.full()) {
        if (!trace->suspend()) {
            DT(3, "*** tex-queue-stall: " << *trace);
        }
        return;
    } else {
        trace->resume();
    }

    // send memory request
    auto trace_data = dynamic_cast<TraceData*>(trace->data);

    uint32_t addr_count = 0;
    for (auto& mem_addr : trace_data->mem_addrs) {
        addr_count += mem_addr.size();
    }

    auto tag = pending_reqs_.allocate({trace, addr_count});

    for (uint32_t t = 0; t < num_threads_; ++t) {
        if (!trace->tmask.test(t))
            continue;

        auto& tcache_req_port = tcache_->CoreReqPorts.at(t);
        for (auto& mem_addr : trace_data->mem_addrs.at(t)) {
            MemReq mem_req;
            mem_req.addr  = mem_addr.addr;
            mem_req.write = (trace->lsu_type == LsuType::STORE);
            mem_req.tag   = tag;
            mem_req.core_id = core_->id();
            mem_req.uuid = trace->uuid;
            tcache_req_port.send(mem_req, config_.address_latency);
            DT(3, "tex-req: addr=" << std::hex << mem_addr.addr << ", tag=" << tag 
                << ", tid=" << t << ", "<< trace);
            ++perf_stats_.reads;
            ++perf_stats_.latency += pending_reqs_.size();
        }
    }

    auto time = simobject_->Input.pop();
    perf_stats_.stalls += (SimPlatform::instance().cycles() - time);
  }

  const PerfStats& perf_stats() const { 
      return perf_stats_; 
  }
};

///////////////////////////////////////////////////////////////////////////////

TexUnit::TexUnit(const SimContext& ctx, const char* name, const Config& config, Core* core) 
  : SimObject<TexUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, config, core)) 
{}

TexUnit::~TexUnit() {
  delete impl_;
}

void TexUnit::reset() {
  impl_->clear();
}

uint32_t TexUnit::read(uint32_t stage, int32_t u, int32_t v, int32_t lod, TraceData* trace_data) {
  return impl_->read(stage, u, v, lod, trace_data);
}

void TexUnit::tick() {
  impl_->tick();
}

const TexUnit::PerfStats& TexUnit::perf_stats() const {
    return impl_->perf_stats();
}