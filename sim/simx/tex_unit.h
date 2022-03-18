#pragma once

#include <simobject.h>
#include <vector>
#include <VX_types.h>
#include "pipeline.h"

namespace vortex {

class Core;

class TexUnit : public SimObject<TexUnit> {
public:
    struct Config {
        uint32_t address_latency;
        uint32_t sampler_latency;
    };

    struct TraceData : public ITraceData {
        std::vector<std::vector<mem_addr_size_t>> mem_addrs;
    };

    class DCRS {
    private:
        std::array<std::array<uint32_t, (DCR_TEX_STATE_COUNT-1)>, TEX_STAGE_COUNT> states_;
        uint32_t stage_;

    public:
        DCRS() {
            this->clear();
        }
    
        void clear() {
            stage_ = 0;
            for (auto& states : states_) {
                for (auto& state : states) {
                    state = 0;
                }
            }
        }

        uint32_t read(uint32_t addr) const {
            if (addr == DCR_TEX_STAGE) {
                return stage_;
            }
            uint32_t state = DCR_TEX_STATE(addr-1);
            return states_.at(stage_).at(state);
        }
    
        void write(uint32_t addr, uint32_t value) {
            if (addr == DCR_TEX_STAGE) {
                stage_ = value;
                return;
            }
            uint32_t state = DCR_TEX_STATE(addr-1);
            states_.at(stage_).at(state) = value;
        }
    };

    struct PerfStats {
        uint64_t stalls;
        uint64_t reads;
        uint64_t latency;

        PerfStats() 
            : stalls(0)
            , reads(0)
            , latency(0)
        {}
    };

    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    TexUnit(const SimContext& ctx, const char* name, const Config& config, Core* core);    
    ~TexUnit();

    void reset();

    uint32_t csr_read(uint32_t addr);

    void csr_write(uint32_t addr, uint32_t value);

    uint32_t read(int32_t u, int32_t v, int32_t lod, TraceData* trace_data);
    
    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}