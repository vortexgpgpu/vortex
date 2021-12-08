#pragma once

#include <simobject.h>
#include "types.h"
#include <vector>

namespace vortex {

class MemSim : public SimObject<MemSim>{
public:
    struct Config {        
        uint32_t channels;      
        uint32_t num_cores;
    };

    struct PerfStats {
        uint64_t reads;
        uint64_t writes;

        PerfStats() 
            : reads(0)
            , writes(0)
        {}
    };

    SimPort<MemReq> MemReqPort;
    SimPort<MemRsp> MemRspPort;

    MemSim(const SimContext& ctx, const char* name, const Config& config);
    ~MemSim();

    void reset();

    void tick();

    const PerfStats& perf_stats() const;
    
private:
    class Impl;
    Impl* impl_;
};

};