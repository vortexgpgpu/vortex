#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "raster_unit.h"
#include "pipeline.h"

namespace vortex {

class Core;
class RasterUnit;

class RasterSrv : public SimObject<RasterSrv> {
public:
    struct PerfStats {
        uint64_t stalls;

        PerfStats() 
            : stalls(0)
        {}
    };
    
    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    RasterSrv(const SimContext& ctx, 
              const char* name,  
              Core* core,
              RasterUnit::Ptr raster_unit);    

    ~RasterSrv();

    void reset();

    uint32_t csr_read(uint32_t wid, uint32_t tid, uint32_t addr);

    void csr_write(uint32_t wid, uint32_t tid, uint32_t addr, uint32_t value);

    uint32_t fetch(uint32_t wid, uint32_t tid);

    int32_t interpolate(uint32_t wid, uint32_t tid, int32_t a, int32_t b, int32_t c);

    void tick();

    const PerfStats& perf_stats() const;
    
private:

    class Impl;
    Impl* impl_;
};

}