#pragma once

#include <simobject.h>
#include "pipeline.h"

namespace vortex {

class Core;

class RopUnit : public SimObject<RopUnit> {
public:
    struct PerfStats {
        uint64_t reads;

        PerfStats() 
            : reads(0)
        {}
    };

    SimPort<pipeline_trace_t*> Input;

    RopUnit(const SimContext& ctx, const char* name, Core* core);    
    ~RopUnit();

    void reset();

    uint32_t csr_read(uint32_t addr);
  
    void csr_write(uint32_t addr, uint32_t value);

    void write(uint32_t x, uint32_t y, uint32_t z, uint32_t color);

    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}