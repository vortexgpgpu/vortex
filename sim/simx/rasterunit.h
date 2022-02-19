#pragma once

#include <simobject.h>
#include "pipeline.h"

namespace vortex {

class Core;

class RasterUnit : public SimObject<RasterUnit> {
public:
    struct raster_quad_t {
        uint32_t x;
        uint32_t y;
        uint32_t mask;
        uint32_t pidx;
    };
    
    struct PerfStats {
        uint64_t reads;

        PerfStats() 
            : reads(0)
        {}
    };

    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    RasterUnit(const SimContext& ctx, const char* name, Core* core);    
    ~RasterUnit();

    void reset();

    uint32_t csr_read(uint32_t addr);
  
    void csr_write(uint32_t addr, uint32_t value);

    bool pop(raster_quad_t* quad);

    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}