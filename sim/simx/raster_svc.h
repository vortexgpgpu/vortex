#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "raster_unit.h"
#include "pipeline.h"

namespace vortex {

class Core;
class RasterUnit;

class RasterSvc : public SimObject<RasterSvc> {
public:
    struct PerfStats {
        uint64_t stalls;

        PerfStats() 
            : stalls(0)
        {}
    };
    
    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    RasterSvc(const SimContext& ctx, 
              const char* name,  
              Core* core,
              RasterUnit::Ptr raster_unit);    

    ~RasterSvc();

    void reset();

    uint32_t csr_read(uint32_t wid, uint32_t tid, uint32_t addr);

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

    uint32_t fetch(uint32_t wid, uint32_t tid);

    void tick();

    const PerfStats& perf_stats() const;
    
private:

    class Impl;
    Impl* impl_;
};

}