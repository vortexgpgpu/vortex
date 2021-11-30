#include "memsim.h"
#include <vector>
#include <queue>
#include "constants.h"

using namespace vortex;

class MemSim::Impl {
private:
    MemSim* simobject_;
    uint32_t num_banks_;
    uint32_t latency_;
    PerfStats perf_stats_;

public:
    Impl(MemSim* simobject, uint32_t num_banks, uint32_t latency) 
        : simobject_(simobject)
        , num_banks_(num_banks)
        , latency_(latency)  
    {}

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

    void step(uint64_t /*cycle*/) {
        for (uint32_t i = 0, n = num_banks_; i < n; ++i) {
            auto& mem_req_port = simobject_->MemReqPorts.at(i); 
            if (mem_req_port.empty())
                continue;
            auto& mem_req = mem_req_port.front();
            if (!mem_req.write) {
                MemRsp mem_rsp;
                mem_rsp.tag = mem_req.tag;
                simobject_->MemRspPorts.at(i).send(mem_rsp, latency_);
                ++perf_stats_.reads;
            } else {
                ++perf_stats_.writes;
            }
            mem_req_port.pop();
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, 
               uint32_t num_banks,
               uint32_t latency) 
    : SimObject<MemSim>(ctx, "MemSim")
    , MemReqPorts(num_banks, this) 
    , MemRspPorts(num_banks, this)
    , impl_(new Impl(this, num_banks, latency))
{}

MemSim::~MemSim() {
    delete impl_;
}

void MemSim::step(uint64_t cycle) {
    impl_->step(cycle);
}