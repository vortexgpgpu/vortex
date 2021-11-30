#pragma once

#include <simobject.h>
#include "types.h"
#include <vector>

namespace vortex {

class MemSim : public SimObject<MemSim>{
public:
    struct PerfStats {
        uint64_t reads;
        uint64_t writes;

        PerfStats() 
            : reads(0)
            , writes(0)
        {}
    };

    std::vector<SimPort<MemReq>> MemReqPorts;
    std::vector<SimPort<MemRsp>> MemRspPorts;

    MemSim(const SimContext& ctx, uint32_t num_banks, uint32_t latency);
    ~MemSim();

    void step(uint64_t cycle);

    const PerfStats& perf_stats() const;
    
private:
    class Impl;
    Impl* impl_;
};

};