#include "tex_unit.h"
#include "mem.h"
#include <VX_config.h>

using namespace vortex;
using namespace cocogfx;

class TexUnit::Impl {
public:

  Impl(TexUnit* simobject, const Arch &arch, const DCRS& dcrs, const Config& config)
    : simobject_(simobject)
    , config_(config)
    , dcrs_(dcrs)
    , sampler_(memoryCB, this)
    , mem_(nullptr)
    , num_threads_(arch.num_threads())
    , pending_reqs_(TEX_MEM_QUEUE_SIZE)
  {
    this->clear();
  }

  ~Impl() {}

  void clear() {
    sampler_.configure(dcrs_);
    pending_reqs_.clear();
  }

  void tick() {
    // handle memory response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& tcache_rsp_port = simobject_->MemRsps.at(t);
        if (tcache_rsp_port.empty())
            continue;
        auto& mem_rsp = tcache_rsp_port.front();
        auto& entry = pending_reqs_.at(mem_rsp.tag);  
        auto trace = entry.trace;
        DT(3, simobject_->name() << "-tex-rsp: tag=" << mem_rsp.tag << ", tid=" << t << ", " << *trace);  
        assert(entry.count);
        --entry.count; // track remaining addresses 
        if (0 == entry.count) {
            simobject_->Output.send(trace, config_.sampler_latency);
            pending_reqs_.release(mem_rsp.tag);
        }   
        tcache_rsp_port.pop();
    }

    for (int i = 0, n = pending_reqs_.size(); i < n; ++i) {
      if (pending_reqs_.contains(i))
        perf_stats_.latency += pending_reqs_.at(i).count;
    }

    // check input queue
    if (simobject_->Input.empty())
        return;

    auto trace = simobject_->Input.front();

    // check pending queue capacity    
    if (pending_reqs_.full()) {
        if (!trace->log_once(true)) {
            DT(3, "*** " << simobject_->name() << "-tex-queue-stall: " << *trace);
        }
        ++perf_stats_.stalls;
        return;
    } else {
        trace->log_once(false);
    }

    // send memory request
    auto trace_data = std::dynamic_pointer_cast<TraceData>(trace->data);

    uint32_t addr_count = 0;
    for (auto& mem_addr : trace_data->mem_addrs) {
        addr_count += mem_addr.size();
    }
    if (addr_count != 0) {
      auto tag = pending_reqs_.allocate({trace, addr_count});
      for (uint32_t t = 0; t < num_threads_; ++t) {
          if (!trace->tmask.test(t))
              continue;

          auto& tcache_req_port = simobject_->MemReqs.at(t);
          for (auto& mem_addr : trace_data->mem_addrs.at(t)) {
              MemReq mem_req;
              mem_req.addr  = mem_addr.addr;
              mem_req.write = (trace->lsu_type == LsuType::STORE);
              mem_req.tag   = tag;
              mem_req.cid   = trace->cid;
              mem_req.uuid  = trace->uuid;
              tcache_req_port.send(mem_req, config_.address_latency);
              DT(3, simobject_->name() << "-tex-req: addr=" << std::hex << mem_addr.addr << ", tag=" << tag 
                  << ", tid=" << t << ", "<< trace);
              ++perf_stats_.reads;
          }
      }
    } else {
      simobject_->Output.send(trace, 1);
    }

    simobject_->Input.pop();
  } 

  uint32_t read(uint32_t cid, uint32_t wid, uint32_t tid,
                uint32_t stage, int32_t u, int32_t v, uint32_t lod, 
                const CSRs& csrs, TraceData::Ptr trace_data) {  
    __unused (cid, wid, csrs);
    mem_addrs_ = &trace_data->mem_addrs.at(tid);
    return sampler_.read(stage, u, v, lod);
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  const PerfStats& perf_stats() const { 
      return perf_stats_; 
  }

private:

  void texture_read(
    uint32_t* out,
    const uint64_t* addr,    
    uint32_t stride,
    uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
      mem_->read(&out[i], addr[i], stride);
      mem_addrs_->push_back({addr[i], stride});
    }
  }

  static void memoryCB(    
    uint32_t* out,
    const uint64_t* addr,    
    uint32_t stride,
    uint32_t size,
    void* cb_arg) {
    reinterpret_cast<Impl*>(cb_arg)->texture_read(out, addr, stride, size);
  }

  struct pending_req_t {
    pipeline_trace_t* trace;
    uint32_t count;
  };

  TexUnit* simobject_;
  Config config_;
  const DCRS& dcrs_;    
  graphics::TextureSampler sampler_;
  std::vector<mem_addr_size_t>* mem_addrs_;
  RAM* mem_;
  uint32_t num_threads_;
  HashTable<pending_req_t> pending_reqs_;
  PerfStats perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TexUnit::TexUnit(const SimContext& ctx, 
                 const char* name,
                 const Arch &arch, 
                 const DCRS& dcrs,   
                 const Config& config)
  : SimObject<TexUnit>(ctx, name)
  , MemReqs(arch.num_threads(), this)
  , MemRsps(arch.num_threads(), this)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, arch, dcrs, config)) 
{}

TexUnit::~TexUnit() {
  delete impl_;
}

void TexUnit::reset() {
  impl_->clear();
}

void TexUnit::tick() {
  impl_->tick();
}

void TexUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

uint32_t TexUnit::read(uint32_t cid, uint32_t wid, uint32_t tid,
                       uint32_t stage, int32_t u, int32_t v, uint32_t lod, 
                       const CSRs& csrs, TraceData::Ptr trace_data) {
  return impl_->read(cid, wid, tid, stage, u, v, lod, csrs, trace_data);
}

const TexUnit::PerfStats& TexUnit::perf_stats() const {
    return impl_->perf_stats();
}
