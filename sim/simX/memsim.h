#pragma once

#include <simobject.h>
#include <vector>
#include <list>

namespace vortex {

struct MemReq {
    uint64_t addr;
    uint32_t tag;
    bool write;
};

struct MemRsp {
    uint32_t tag;
};

class MemSim : public SimObject<MemSim>{
private:
    class Impl;
    Impl* impl_;

public:

    MemSim(const SimContext& ctx, uint32_t num_inputs, uint32_t latency);
    ~MemSim();

    void step(uint64_t cycle);

    std::vector<SlavePort<MemReq>>  MemReqPorts;
    std::vector<MasterPort<MemRsp>> MemRspPorts;
};

};