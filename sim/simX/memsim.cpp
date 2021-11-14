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

public:
    Impl(MemSim* simobject, uint32_t num_banks, uint32_t latency) 
        : simobject_(simobject)
        , num_banks_(num_banks)
        , latency_(latency)  
    {}

    void step(uint64_t /*cycle*/) {
        for (uint32_t i = 0, n = num_banks_; i < n; ++i) {
            MemReq mem_req;     
            if (!simobject_->MemReqPorts.at(i).read(&mem_req))
                continue;
            if (!mem_req.write) {
                MemRsp mem_rsp;
                mem_rsp.tag = mem_req.tag;
                simobject_->MemRspPorts.at(i).send(mem_rsp, latency_);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, 
               uint32_t num_banks,
               uint32_t latency) 
    : SimObject<MemSim>(ctx, "MemSim")
    , impl_(new Impl(this, num_banks, latency))
    , MemReqPorts(num_banks, this) 
    , MemRspPorts(num_banks, this)
{}

MemSim::~MemSim() {
    delete impl_;
}

void MemSim::step(uint64_t cycle) {
    impl_->step(cycle);
}