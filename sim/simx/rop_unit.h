#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"

namespace vortex {

class RAM;
class Core;

class RopUnit : public SimObject<RopUnit> {
public:
    struct PerfStats {
        uint64_t reads;

        PerfStats() 
            : reads(0)
        {}
    };

    class DCRS {
    private:
        std::array<uint32_t, ROP_STATE_COUNT> states_;

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

        uint32_t read(uint32_t addr) const {
            uint32_t state = DCR_ROP_STATE(addr);
            return states_.at(state);
        }
    
        void write(uint32_t addr, uint32_t value) {
            uint32_t state = DCR_ROP_STATE(addr);
            states_.at(state) = value;
        }
    };

    SimPort<pipeline_trace_t*> MemReq;
    SimPort<pipeline_trace_t*> MemRsp;

    RopUnit(const SimContext& ctx, 
            const char* name,  
            const Arch &arch, 
            const DCRS& dcrs);    

    ~RopUnit();

    void attach_ram(RAM* mem);

    void reset();

    void write(uint32_t x, uint32_t y, uint32_t mask, uint32_t face, uint32_t color, uint32_t depth);

    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}