#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "rop_unit.h"
#include "pipeline.h"

namespace vortex {

class Core;
class RopUnit;

class RopSrv : public SimObject<RopSrv> {
public:
    struct PerfStats {
        uint64_t stalls;

        PerfStats() 
            : stalls(0)
        {}
    };
    
    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    RopSrv(const SimContext& ctx, 
           const char* name,  
           Core* core,
           RopUnit::Ptr rop_unit);    

    ~RopSrv();

    void reset();

    uint32_t csr_read(uint32_t wid, uint32_t tid, uint32_t addr);
    
    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

    void write(uint32_t x, uint32_t y , uint32_t mask, uint32_t color, uint32_t depth);

    void tick();

    const PerfStats& perf_stats() const;
    
private:

    class Impl;
    Impl* impl_;
};

}