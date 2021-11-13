#include "memsim.h"
#include <vector>
#include <queue>
#include "constants.h"

using namespace vortex;

class MemSim::Impl {
private:
    MemSim* simobject_;
    std::vector<std::queue<MemReq>> inputs_;
    uint32_t latency_;

public:
    Impl(MemSim* simobject, uint32_t num_banks, uint32_t latency) 
        : simobject_(simobject)
        , inputs_(num_banks)
        , latency_(latency)  
    {}

    void handleMemRequest(const MemReq& mem_req, uint32_t port_id) {
        inputs_.at(port_id).push(mem_req);        
    }

    void step(uint64_t /*cycle*/) {
        for (uint32_t i = 0, n = inputs_.size(); i < n; ++i) {
            auto& queue = inputs_.at(i);            
            if (queue.empty())
                continue;
            auto& entry = queue.front();
            if (!entry.write) {
                MemRsp mem_rsp;
                mem_rsp.tag = entry.tag;
                simobject_->MemRspPorts.at(i).send(mem_rsp, latency_);
            }
            queue.pop();
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, 
               uint32_t num_banks,
               uint32_t latency) 
    : SimObject<MemSim>(ctx, "MemSim")
    , impl_(new Impl(this, num_banks, latency))
    , MemReqPorts(num_banks, {this, impl_, &Impl::handleMemRequest}) 
    , MemRspPorts(num_banks, this)
{}

MemSim::~MemSim() {
    delete impl_;
}

void MemSim::step(uint64_t cycle) {
    impl_->step(cycle);
}