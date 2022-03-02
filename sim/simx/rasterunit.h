#pragma once

#include <simobject.h>
#include <VX_types.h>
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

    class DCRS {
    private:
        std::array<uint32_t, RASTER_STATE_COUNT> states_;

    public:
        DCRS() {
            this->clear();
        }
    
        void clear() {
            for (auto& state : states_) {
                state = 0;
            }
        }

        uint32_t at(uint32_t state) const {
            return states_.at(state);
        }

        uint32_t read(uint32_t addr) {
            uint32_t state = DCR_RASTER_STATE(addr);
            return states_.at(state);
        }
    
        void write(uint32_t addr, uint32_t value) {
            uint32_t state = DCR_RASTER_STATE(addr);
            states_.at(state) = value;
        }
    };

    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    RasterUnit(const SimContext& ctx, const char* name, Core* core);    
    ~RasterUnit();

    void reset();

    uint32_t dcr_read(uint32_t addr);
  
    void dcr_write(uint32_t addr, uint32_t value);

    bool pop(raster_quad_t* quad);

    int32_t interpolate(uint32_t quad, int32_t a, int32_t b, int32_t c);

    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}